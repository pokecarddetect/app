# 🚀 Pokémon Card Authentication App - Deployment Guide

Kompletní návod pro nasazení Pokémon Card Authentication aplikace na různých cloud platformách.

## 📋 Příprava před nasazením

### 1. Stáhnutí aplikace z Replit
```bash
# V Replit:
# 1. Klikněte na Files (vlevo)
# 2. Klikněte na tři tečky (...) nebo hamburger menu
# 3. Vyberte "Download as zip"
# 4. Rozbalte soubory na svém počítači
```

### 2. Alternativní způsob - Git repository
```bash
git clone https://github.com/your-username/pokemon-card-app.git
cd pokemon-card-app
```

## 🌐 Railway Deployment (Doporučeno)

Railway je nejjednodušší způsob nasazení s automatickou PostgreSQL databází.

### Rychlé nasazení:
1. **Připojte GitHub repository**
   - Nahrajte kód na GitHub
   - Přihlaste se na railway.app
   - Klikněte "New Project" → "Deploy from GitHub repo"

2. **Konfigurace proběhne automaticky**
   - Railway detekuje `railway.toml` a `Procfile`
   - Automaticky vytvoří PostgreSQL databázi
   - Nastaví environment proměnné

3. **Nastavte pouze SECRET_KEY**
   ```bash
   # V Railway dashboard → Variables
   SESSION_SECRET=your-super-secret-key-change-this
   ```

4. **Deploy!** 
   - Railway automaticky buildy a deploy aplikaci
   - Přístup na `https://your-app.railway.app`

### Manuální nasazení přes CLI:
```bash
# Instalace Railway CLI
npm install -g @railway/cli
railway login

# V složce s aplikací
railway create
railway add postgresql  # Přidá PostgreSQL databázi
railway deploy

# Nastavení secrets
railway variables set SESSION_SECRET=your-secret-key
```

## 🌊 Heroku Deployment

### Příprava:
```bash
# Instalace Heroku CLI
# Stáhněte z: https://devcenter.heroku.com/articles/heroku-cli

heroku login
heroku create your-pokemon-app-name
```

### Databáze a konfigurace:
```bash
# Přidání PostgreSQL
heroku addons:create heroku-postgresql:hobby-dev

# Nastavení environment proměnných
heroku config:set SESSION_SECRET=your-super-secret-key
heroku config:set FLASK_ENV=production
heroku config:set DEPLOYMENT_MODE=true
```

### Deploy:
```bash
git init
git add .
git commit -m "Initial commit"
git push heroku main

# Kontrola logů
heroku logs --tail
```

## 🌊 DigitalOcean App Platform

### Přes Web Interface:
1. **Vytvořte novou App** na cloud.digitalocean.com
2. **Připojte GitHub repository**  
3. **Konfigurace se detekuje automaticky** z `Procfile`
4. **Přidejte PostgreSQL databázi** v Resources sekci
5. **Nastavte environment proměnné:**
   ```
   SESSION_SECRET=your-secret-key
   FLASK_ENV=production
   DEPLOYMENT_MODE=true
   ```

### Přes CLI:
```bash
# Instalace doctl CLI
# Stáhněte z: https://docs.digitalocean.com/reference/doctl/

doctl apps create --spec .do/app.yaml
```

## 🐳 Docker Deployment

### Lokální testování:
```bash
# Build a spuštění s databází
docker-compose up --build

# Aplikace běží na http://localhost:5000
```

### Produkční nasazení:
```bash
# Build image
docker build -t pokemon-card-app .

# Run s externí databází
docker run -p 5000:5000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/dbname \
  -e SESSION_SECRET=your-secret-key \
  -e DEPLOYMENT_MODE=true \
  pokemon-card-app
```

### Docker Hub deployment:
```bash
# Push to Docker Hub
docker tag pokemon-card-app your-username/pokemon-card-app
docker push your-username/pokemon-card-app

# Deploy na cloud provider s Docker podporou
```

## ⚡ Vercel (Serverless)

### Příprava:
```bash
# Instalace Vercel CLI
npm install -g vercel

# V aplikační složce
vercel

# Následujte setup wizard
```

### Konfigurace:
- Vercel automaticky detekuje `vercel.json`
- Nastavte environment proměnné v Vercel dashboard
- **Poznámka:** Potřebujete externí PostgreSQL (např. Supabase, PlanetScale)

## 🛠️ Environment proměnné

### Vyžadované pro všechny platformy:
```bash
DATABASE_URL=postgresql://user:password@host:5432/database_name
SESSION_SECRET=your-super-secret-key-change-in-production
```

### Doporučené pro produkci:
```bash
FLASK_ENV=production
DEPLOYMENT_MODE=true
PYTHONPATH=.
```

### Volitelné:
```bash
MAX_CONTENT_LENGTH=16777216  # 16MB file upload limit
UPLOAD_FOLDER=uploads
```

## 🗄️ Databáze setup

### Automatické (Railway, Heroku):
- PostgreSQL se vytvoří automaticky
- DATABASE_URL se nastaví automaticky  
- Aplikace vytvoří tabulky při prvním spuštění

### Manuální setup:
```sql
-- Připojte se k PostgreSQL databázi
CREATE DATABASE pokemon_cards;

-- Aplikace vytvoří tabulky automaticky při prvním spuštění
```

## 📈 Performance optimalizace

### Pro malé aplikace:
- **Railway:** Hobby plan ($5/měsíc)
- **Heroku:** Hobby dyno ($7/měsíc)
- **DigitalOcean:** Basic droplet ($4/měsíc)

### Pro větší provoz:
- Zvyšte počet workers: `--workers 2`
- Přidejte Redis cache
- Použijte CDN pro statické soubory

## 🔧 Troubleshooting

### Problém: "Port již používán"
```bash
# Změňte port v konfiguraci nebo zabijte proces
lsof -ti:5000 | xargs kill -9
```

### Problém: "Database connection failed"
```bash
# Zkontrolujte DATABASE_URL
echo $DATABASE_URL

# Otestujte připojení
psql $DATABASE_URL -c "SELECT 1;"
```

### Problém: "TensorFlow errors"
```bash
# Nastavte DEPLOYMENT_MODE
export DEPLOYMENT_MODE=true

# Aplikace použije lightweight fallback model
```

### Problém: "OCR nefunguje"
```bash
# Nainstalujte Tesseract (buildpack pro Heroku/Railway)
# Pro Ubuntu/Debian:
apt-get install tesseract-ocr

# Pro Docker - již zahrnuto v Dockerfile
```

## 📱 Mobile & Desktop alternativy

### PWA (Progressive Web App):
- Aplikace funguje jako native app na mobilu
- Automatická instalace přes browser

### Desktop aplikace:
```bash
# Electron wrapper (volitelné)
npm install -g electron
electron .
```

## 🎯 Doporučené workflow

1. **Vývoj:** Replit (rychlý prototyping)
2. **Testování:** Docker locally (`docker-compose up`)
3. **Staging:** Railway (rychlé deployment)
4. **Produkce:** DigitalOcean/AWS (škálovatelnost)

## 💡 Pro-tips

- **Používejte PostgreSQL** i pro development (SQLite jen pro testing)
- **Nastavte monitoring** (Sentry, LogRocket)
- **Implementujte backupy** databáze
- **Použijte CDN** pro rychlé načítání (Cloudflare)
- **Aktivujte HTTPS** (automaticky na většině platforem)

---

## 🎉 Hotovo!

Vaše Pokémon Card Authentication aplikace je nyní připravena pro nasazení na produkci. Aplikace dokáže zpracovat tisíce karet denně s vysokou přesností AI detekce.

### Support
- Issues: GitHub repository  
- Discord: [Pokémon Card Developers]
- Email: support@pokemonapp.com