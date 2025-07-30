import jwt
import os
import uuid
from functools import wraps
from urllib.parse import urlencode

from flask import g, session, redirect, request, render_template, url_for
from flask_dance.consumer import (
    OAuth2ConsumerBlueprint,
    oauth_authorized,
    oauth_error,
)
from flask_dance.consumer.storage import BaseStorage
from flask_login import LoginManager, login_user, logout_user, current_user
from oauthlib.oauth2.rfc6749.errors import InvalidGrantError
from sqlalchemy.exc import NoResultFound
from werkzeug.local import LocalProxy

from app import app, db
from models import OAuth, User

login_manager = LoginManager(app)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)


class UserSessionStorage(BaseStorage):

    def get(self, blueprint):
        try:
            token = db.session.query(OAuth).filter_by(
                user_id=current_user.get_id(),
                browser_session_key=g.browser_session_key,
                provider=blueprint.name,
            ).one().token
        except NoResultFound:
            token = None
        return token

    def set(self, blueprint, token):
        db.session.query(OAuth).filter_by(
            user_id=current_user.get_id(),
            browser_session_key=g.browser_session_key,
            provider=blueprint.name,
        ).delete()
        new_model = OAuth()
        new_model.user_id = current_user.get_id()
        new_model.browser_session_key = g.browser_session_key
        new_model.provider = blueprint.name
        new_model.token = token
        db.session.add(new_model)
        db.session.commit()

    def delete(self, blueprint):
        db.session.query(OAuth).filter_by(
            user_id=current_user.get_id(),
            browser_session_key=g.browser_session_key,
            provider=blueprint.name).delete()
        db.session.commit()


def make_replit_blueprint():
    try:
        repl_id = os.environ['REPL_ID']
    except KeyError:
        raise SystemExit("the REPL_ID environment variable must be set")

    issuer_url = os.environ.get('ISSUER_URL', "https://replit.com/oidc")

    replit_bp = OAuth2ConsumerBlueprint(
        "replit_auth",
        __name__,
        client_id=repl_id,
        client_secret=None,
        base_url=issuer_url,
        authorization_url_params={
            "prompt": "login consent",
        },
        token_url=issuer_url + "/token",
        token_url_params={
            "auth": (),
            "include_client_id": True,
        },
        auto_refresh_url=issuer_url + "/token",
        auto_refresh_kwargs={
            "client_id": repl_id,
        },
        authorization_url=issuer_url + "/auth",
        use_pkce=True,
        code_challenge_method="S256",
        scope=["openid", "profile", "email", "offline_access"],
        storage=UserSessionStorage(),
    )

    @replit_bp.before_app_request
    def set_applocal_session():
        if '_browser_session_key' not in session:
            session['_browser_session_key'] = uuid.uuid4().hex
        session.modified = True
        g.browser_session_key = session['_browser_session_key']
        g.flask_dance_replit = replit_bp.session

    @replit_bp.route("/logout")
    def logout():
        del replit_bp.token
        logout_user()

        end_session_endpoint = issuer_url + "/session/end"
        encoded_params = urlencode({
            "client_id":
            repl_id,
            "post_logout_redirect_uri":
            request.url_root,
        })
        logout_url = f"{end_session_endpoint}?{encoded_params}"

        return redirect(logout_url)

    @replit_bp.route("/error")
    def error():
        return render_template("403.html"), 403

    return replit_bp


def verify_jwt_with_jwks(id_token, issuer_url, audience):
    """
    Verify JWT token using JWKS (JSON Web Key Set) from the issuer.
    This provides proper cryptographic verification of the JWT signature.
    """
    import requests
    from jwcrypto import jwk, jwt as jwcrypto_jwt
    from jwt.exceptions import InvalidTokenError
    import json
    
    try:
        # Get OIDC configuration
        oidc_config_url = f"{issuer_url}/.well-known/openid_configuration"
        oidc_response = requests.get(oidc_config_url, timeout=10)
        oidc_response.raise_for_status()
        oidc_config = oidc_response.json()
        
        # Get JWKS endpoint
        jwks_uri = oidc_config.get('jwks_uri')
        if not jwks_uri:
            raise ValueError("JWKS URI not found in OIDC configuration")
        
        # Fetch JWKS
        jwks_response = requests.get(jwks_uri, timeout=10)
        jwks_response.raise_for_status()
        jwks_data = jwks_response.json()
        
        # Create JWK set from the JWKS data
        jwks = jwk.JWKSet.from_json(json.dumps(jwks_data))
        
        # Create JWT object and verify
        jwt_token = jwcrypto_jwt.JWT()
        jwt_token.deserialize(id_token)
        
        # Find the correct key for verification
        header = jwt_token.header
        kid = json.loads(header).get('kid') if header else None
        
        # Find matching key in JWKS
        signing_key = None
        for key_data in jwks_data.get('keys', []):
            if key_data.get('kid') == kid:
                signing_key = jwk.JWK.from_json(json.dumps(key_data))
                break
        
        if not signing_key:
            raise ValueError(f"No matching key found for kid: {kid}")
        
        # Verify signature using the specific key
        jwt_token.verify(signing_key)
        
        # Extract and validate claims
        claims = json.loads(jwt_token.claims)
        
        # Verify critical claims
        if claims.get('aud') != audience:
            raise InvalidTokenError("Audience mismatch")
        if claims.get('iss') != issuer_url:
            raise InvalidTokenError("Issuer mismatch")
        
        # Check expiration
        import time
        if claims.get('exp', 0) < time.time():
            raise InvalidTokenError("Token expired")
            
        return claims
        
    except Exception as e:
        app.logger.error(f"JWT verification failed: {str(e)}")
        raise InvalidTokenError(f"JWT verification failed: {str(e)}")


def save_user(user_claims):
    user = User()
    user.id = user_claims['sub']
    user.email = user_claims.get('email')
    user.first_name = user_claims.get('first_name')
    user.last_name = user_claims.get('last_name')
    user.profile_image_url = user_claims.get('profile_image_url')
    merged_user = db.session.merge(user)
    db.session.commit()
    return merged_user


@oauth_authorized.connect
def logged_in(blueprint, token):
    from urllib.parse import urlparse
    from flask import request
    import requests
    from jwt.exceptions import InvalidTokenError
    
    try:
        # Validate that we have a properly structured token from OAuth flow
        if not token or 'id_token' not in token:
            raise ValueError("Invalid token structure received from OAuth flow")
        
        # Get issuer URL for validation
        issuer_url = os.environ.get('ISSUER_URL', "https://replit.com/oidc")
        repl_id = os.environ.get('REPL_ID')
        
        if not repl_id:
            raise ValueError("REPL_ID environment variable not set")
        
        # Properly verify JWT signature using JWKS from Replit's OIDC endpoint
        user_claims = verify_jwt_with_jwks(token['id_token'], issuer_url, repl_id)
        
        # Additional validation of critical claims
        if not user_claims.get('sub'):
            raise ValueError("Token missing required 'sub' claim")
            
        if user_claims.get('aud') != repl_id:
            raise ValueError("Token audience mismatch")
            
        if user_claims.get('iss') != issuer_url:
            raise ValueError("Token issuer mismatch")
        
        user = save_user(user_claims)
        login_user(user)
        blueprint.token = token
        next_url = session.pop("next_url", None)
        if next_url is not None:
            # Validate next_url to prevent open redirect attacks
            parsed_next = urlparse(next_url)
            parsed_request = urlparse(request.url)
            # Only allow redirects to same host
            if parsed_next.netloc == parsed_request.netloc or not parsed_next.netloc:
                return redirect(next_url)
                
    except (InvalidTokenError, ValueError) as e:
        # Log the error and redirect to error page for security
        app.logger.error(f"JWT validation failed: {str(e)}")
        return redirect(url_for('replit_auth.error'))


@oauth_error.connect
def handle_error(blueprint, error, error_description=None, error_uri=None):
    return redirect(url_for('replit_auth.error'))


def require_login(f):

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            session["next_url"] = get_next_navigation_url(request)
            return redirect(url_for('replit_auth.login'))

        expires_in = replit.token.get('expires_in', 0)
        if expires_in < 0:
            issuer_url = os.environ.get('ISSUER_URL', "https://replit.com/oidc")
            refresh_token_url = issuer_url + "/token"
            try:
                token = replit.refresh_token(token_url=refresh_token_url,
                                             client_id=os.environ['REPL_ID'])
            except InvalidGrantError:
                # If the refresh token is invalid, the users needs to re-login.
                session["next_url"] = get_next_navigation_url(request)
                return redirect(url_for('replit_auth.login'))
            replit.token_updater(token)

        return f(*args, **kwargs)

    return decorated_function


def get_next_navigation_url(request):
    from urllib.parse import urlparse
    
    is_navigation_url = request.headers.get(
        'Sec-Fetch-Mode') == 'navigate' and request.headers.get(
            'Sec-Fetch-Dest') == 'document'
    if is_navigation_url:
        return request.url
    
    # Validate referrer to prevent open redirect attacks
    referrer = request.referrer
    if referrer:
        parsed_referrer = urlparse(referrer)
        parsed_request = urlparse(request.url)
        # Only allow redirects to same host
        if parsed_referrer.netloc == parsed_request.netloc:
            return referrer
    
    return request.url


replit = LocalProxy(lambda: g.flask_dance_replit)