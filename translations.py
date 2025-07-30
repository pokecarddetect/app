# Multi-language support for Pokémon Card Detector
# Podpora více jazyků pro detektor Pokémon karet

TRANSLATIONS = {
    'en': {
        # Navigation
        'pokemon_card_detector': 'Pokémon Card Detector',
        'upload_card': 'Upload Card',
        'my_analyses': 'My Analyses',
        'certificates': 'Certificates',
        'ai_training': 'AI Training',
        'feedback': 'Feedback',
        'login': 'Login',
        'logout': 'Logout',
        'language': 'Language',
        
        # Main page
        'welcome_title': 'AI-Powered Pokémon Card Authenticity Detection',
        'welcome_subtitle': 'Upload your card images for instant AI analysis with explainable results',
        'upload_front': 'Upload Front Side',
        'upload_back': 'Upload Back Side (Optional)',
        'analyze_card': 'Analyze Card',
        'supported_formats': 'Supported formats: PNG, JPG, JPEG (max. 16MB per file). Back side is optional but improves analysis accuracy.',
        
        # Analysis features
        'ai_classification': 'AI Classification',
        'ai_classification_desc': 'CNN deep learning model analyzes visual features and authenticity markers.',
        'ocr_text_analysis': 'OCR Text Analysis',
        'ocr_text_analysis_desc': 'Detects spelling errors, font inconsistencies, and text anomalies.',
        'visual_similarity': 'Visual Similarity',
        'visual_similarity_desc': 'SSIM and feature matching comparison with official card database.',
        'explainable_ai': 'Explainable AI',
        'explainable_ai_desc': 'Attention heatmaps show where the AI model focused during analysis.',
        'print_quality': 'Print Quality',
        'print_quality_desc': 'Analyzes sharpness, edge strength, and color consistency.',
        'geometric_analysis': 'Geometric Analysis',
        'geometric_analysis_desc': 'Detects shape distortions and dimensional anomalies.',
        
        # Categories
        'original': 'Original',
        'original_desc': 'Official card issued by Pokémon TCG',
        'fake': 'Fake',
        'fake_desc': 'Counterfeit cards with poor print and material quality',
        'proxy': 'Proxy',
        'proxy_desc': 'High quality reproduction for gameplay - not official',
        'custom_art': 'Custom Art',
        'custom_art_desc': 'Fan-made card with custom artwork and design',
        'altered': 'Altered',
        'altered_desc': 'Modified official card (altered art, foiling, etc.)',
        'test_print': 'Test Print',
        'test_print_desc': 'Prototype or test print of a card',
        
        # Results
        'analysis_results': 'Analysis Results',
        'analyze_another_card': 'Analyze Another Card',
        'prediction': 'Prediction',
        'confidence_score': 'Confidence Score',
        'uploaded': 'Uploaded',
        'front_side': 'Front Side',
        'back_side': 'Back Side',
        'back_not_uploaded': 'Back side not uploaded',
        'image_not_available': 'Image not available',
        'categories_explanation': 'Card Categories - Explanation',
        
        # Feedback
        'help_improve_ai': 'Help Improve Our AI Model',
        'feedback_helps': 'Your feedback helps train our system to be more accurate',
        'original_analysis': 'Original Analysis',
        'your_correction': 'Your Correction',
        'correct_classification': 'What is the correct classification?',
        'select_classification': 'Select correct classification',
        'confidence_level': 'How confident are you?',
        'very_confident': 'Very confident',
        'somewhat_confident': 'Somewhat confident',
        'not_sure': 'Not sure',
        'reasoning': 'Reasoning (optional)',
        'explain_decision': 'Please explain why you think this is the correct classification',
        'submit_feedback': 'Submit Feedback',
        'provide_feedback': 'Provide Feedback',
        'view_analysis_history': 'View Analysis History',
        'analyze_another_card': 'Analyze Another Card',
        
        # Messages
        'access_denied_admin': 'Access denied. Only administrators can access the AI training dashboard.',
        'access_denied_retraining': 'Access denied. Only administrators can start AI model retraining.',
        'file_too_large': 'File is too large. Please upload an image smaller than 16MB.',
        'invalid_file': 'Invalid file type. Please upload PNG, JPG, or JPEG images only.',
        'no_file_selected': 'No file selected. Please choose an image to upload.',
        'analysis_saved': 'Analysis saved successfully.',
        'feedback_submitted': 'Thank you for your feedback! It will help improve our AI model.',
        
        # Admin
        'admin_only': 'Admin Only',
        'training_dashboard': 'AI Training Dashboard',
        'retraining_status': 'Retraining Status',
        'start_retraining': 'Start Retraining',
        'feedback_statistics': 'Feedback Statistics',
    },
    
    'cs': {
        # Navigation
        'pokemon_card_detector': 'Detektor Pokémon Karet',
        'upload_card': 'Nahrát Kartu',
        'my_analyses': 'Moje Analýzy',
        'certificates': 'Certifikáty',
        'ai_training': 'AI Trénink',
        'feedback': 'Zpětná Vazba',
        'login': 'Přihlásit',
        'logout': 'Odhlásit',
        'language': 'Jazyk',
        
        # Main page
        'welcome_title': 'AI Detekce Pravosti Pokémon Karet',
        'welcome_subtitle': 'Nahrajte obrázky svých karet pro okamžitou AI analýzu s vysvětlitelnými výsledky',
        'upload_front': 'Nahrát Přední Stranu',
        'upload_back': 'Nahrát Zadní Stranu (Volitelné)',
        'analyze_card': 'Analyzovat Kartu',
        'supported_formats': 'Podporované formáty: PNG, JPG, JPEG (max. 16MB na soubor). Zadní strana není povinná, ale zlepší přesnost analýzy.',
        
        # Analysis features
        'ai_classification': 'AI Klasifikace',
        'ai_classification_desc': 'CNN model hlubokého učení analyzuje vizuální rysy a značky pravosti.',
        'ocr_text_analysis': 'OCR Analýza Textu',
        'ocr_text_analysis_desc': 'Detekuje pravopisné chyby, nekonzistentnost písem a textové anomálie.',
        'visual_similarity': 'Vizuální Podobnost',
        'visual_similarity_desc': 'SSIM a porovnání funkcí s oficiální databází karet.',
        'explainable_ai': 'Vysvětlitelná AI',
        'explainable_ai_desc': 'Tepelné mapy pozornosti ukazují, kde se AI model během analýzy zaměřil.',
        'print_quality': 'Kvalita Tisku',
        'print_quality_desc': 'Analyzuje ostrost, sílu hran a konzistentnost barev.',
        'geometric_analysis': 'Geometrická Analýza',
        'geometric_analysis_desc': 'Detekuje deformace tvaru a rozměrové anomálie.',
        
        # Categories
        'original': 'Originál',
        'original_desc': 'Oficiální karta vydaná společností Pokémon TCG',
        'fake': 'Padělek',
        'fake_desc': 'Padělané karty s nízkou kvalitou tisku a materiálu',
        'proxy': 'Proxy',
        'proxy_desc': 'Kvalitní reprodukce pro hraní - není oficiální',
        'custom_art': 'Vlastní Design',
        'custom_art_desc': 'Fanmade karta s vlastním designem a uměleckým dílem',
        'altered': 'Upravená',
        'altered_desc': 'Upravená oficiální karta (změny v artwork, foil atd.)',
        'test_print': 'Testovací Tisk',
        'test_print_desc': 'Prototyp nebo testovací tisk karty',
        
        # Results
        'analysis_results': 'Výsledky Analýzy',
        'analyze_another_card': 'Analyzovat Další Kartu',
        'prediction': 'Predikce',
        'confidence_score': 'Skóre Spolehlivosti',
        'uploaded': 'Nahráno',
        'front_side': 'Přední Strana',
        'back_side': 'Zadní Strana',
        'back_not_uploaded': 'Zadní strana nebyla nahrána',
        'image_not_available': 'Obrázek není dostupný',
        'categories_explanation': 'Kategorie Karet - Vysvětlení',
        
        # Feedback
        'help_improve_ai': 'Pomozte Vylepšit Náš AI Model',
        'feedback_helps': 'Vaše zpětná vazba pomáhá trénovat náš systém, aby byl přesnější',
        'original_analysis': 'Původní Analýza',
        'your_correction': 'Vaše Oprava',
        'correct_classification': 'Jaká je správná klasifikace?',
        'select_classification': 'Vyberte správnou klasifikaci',
        'confidence_level': 'Jak si jste jisti?',
        'very_confident': 'Velmi si jsem jistý',
        'somewhat_confident': 'Částečně si jsem jistý',
        'not_sure': 'Nejsem si jistý',
        'reasoning': 'Odůvodnění (volitelné)',
        'explain_decision': 'Prosím vysvětlete, proč si myslíte, že je to správná klasifikace',
        'submit_feedback': 'Odeslat Zpětnou Vazbu',
        'provide_feedback': 'Poskytnout Zpětnou Vazbu',
        'view_analysis_history': 'Zobrazit Historii Analýz',
        'analyze_another_card': 'Analyzovat Další Kartu',
        
        # Messages
        'access_denied_admin': 'Přístup odepřen. Pouze administrátoři mohou přistupovat k AI tréninkovému panelu.',
        'access_denied_retraining': 'Přístup odepřen. Pouze administrátoři mohou spustit přetrénování AI modelu.',
        'file_too_large': 'Soubor je příliš velký. Prosím nahrajte obrázek menší než 16MB.',
        'invalid_file': 'Neplatný typ souboru. Prosím nahrajte pouze PNG, JPG nebo JPEG obrázky.',
        'no_file_selected': 'Není vybrán žádný soubor. Prosím vyberte obrázek k nahrání.',
        'analysis_saved': 'Analýza byla úspěšně uložena.',
        'feedback_submitted': 'Děkujeme za zpětnou vazbu! Pomůže vylepšit náš AI model.',
        
        # Admin
        'admin_only': 'Pouze Admin',
        'training_dashboard': 'AI Tréninkový Panel',
        'retraining_status': 'Stav Přetrénování',
        'start_retraining': 'Spustit Přetrénování',
        'feedback_statistics': 'Statistiky Zpětné Vazby',
    }
}

def get_text(key, lang='en'):
    """Get translated text for the given key and language"""
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)

def get_available_languages():
    """Get list of available language codes"""
    return list(TRANSLATIONS.keys())

def get_language_names():
    """Get human-readable language names"""
    return {
        'en': 'English',
        'cs': 'Čeština'
    }