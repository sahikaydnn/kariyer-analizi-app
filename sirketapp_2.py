import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import re
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Kariyer Yolu Öneri Sistemi",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
    }
    .career-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    .company-card {
        background: linear-gradient(135deg, #35688c 0%, #20c997 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
    }
    .service-sector-alert {
        background: linear-gradient(135deg, #a9c6db 0%, #cde7fa 50%, #cde7fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: #a9c6db;
        text-align: center;
        border: 2px solid ##35688c;
    }
</style>
""", unsafe_allow_html=True)

class CareerAnalysisApp:
    def __init__(self):
        # Career paths definitions
        self.career_paths = {
            "Yurt içi vakıf akademisyen": [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
            "Yurt içi devlet akademisyen": [1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
            "Yurt dışı vakıf akademisyen": [0, 1, 1, 1, 1, 0, 1, 1, 0, 0],
            "Yurt dışı devlet akademisyen": [1, 1, 1, 0, 1, 0, 1, 1, 0, 0],
            "Hizmet sektörü çalışma hayatı": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "Yüksek lisans sonrası hizmet sektörü": [1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
            "Üretim sektörü çalışma hayatı": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            "Yüksek lisans sonrası üretim sektörü": [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
            "Start-up": [0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
        }
        
        self.variable_names = [
            "Intercept (Sabit)",
            "Stres Düzeyi", 
            "Kendini Geliştirme",
            "Şirket Kültürü",
            "Yan Haklar",
            "Şirketin Uluslararası Olması",
            "Ekip Bağları",
            "Yurtdışı İmkanları",
            "Maaş",
            "Şirketin Konumu",
            "Yükselme Potansiyeli"
        ]
        
        # Hizmet sektörü kariyer yolları
        self.service_sector_careers = [
            "Hizmet sektörü çalışma hayatı",
            "Yüksek lisans sonrası hizmet sektörü"
        ]
        
        # Şirket verileri (embedded)
        self.company_data = [
            ["Aselsan", [1,0,0,0,0,1,0,1,0,0]],
            ["Opet", [1,1,1,1,0,1,0,1,0,0]],
            ["British American Tobacco", [1,1,1,0,1,1,1,0,1,1]],
            ["L'Oreal", [0,1,1,0,1,1,1,1,0,1]],
            ["Odeabank", [0,1,1,0,0,1,0,0,1,0]],
            ["Koç Sistem", [1,0,0,1,0,0,0,0,0,0]],
            ["EY", [0,1,1,0,1,1,1,0,1,0]],
            ["JTI", [0,1,1,1,1,1,1,1,0,1]],
            ["Aktiftech Technology", [1,1,1,1,1,1,1,1,0,1]],
            ["MetLife", [1,0,1,0,1,1,1,0,0,0]],
            ["Turkish Technic", [1,1,0,1,1,0,1,1,0,0]],
            ["BNP Paribas Cardif", [1,0,1,0,1,0,1,0,1,0]],
            ["PwC", [0,1,1,0,1,1,1,0,0,0]],
            ["Getir", [0,1,1,0,1,1,1,0,1,0]],
            ["Emlak Katılım Bankası", [1,0,0,0,0,1,0,1,0,1]],
            ["TUSAŞ", [1,0,0,1,0,0,0,1,0,0]],
            ["AJet", [1,0,1,1,0,1,0,0,0,0]],
            ["EnerjiSA", [0,1,1,1,0,1,0,0,1,0]],
            ["Bosch Türkiye", [1,0,1,1,1,1,1,1,1,0]],
            ["Papara", [1,1,1,1,0,1,0,1,1,0]],
            ["Turkish Technology", [0,0,0,1,1,1,0,1,0,0]],
            ["Danone", [1,1,1,1,1,0,1,0,1,1]],
            ["Garanti BBVA", [1,1,1,1,1,1,1,1,1,0]],
            ["Garanti Teknoloji", [0,1,1,1,1,1,1,1,0,0]],
            ["Arçelik", [0,1,0,0,1,0,0,0,0,0]],
            ["Schneider Electric", [1,0,1,0,1,0,1,0,0,1]],
            ["QNB", [1,0,0,0,1,1,0,0,1,0]],
            ["Koçtaş", [1,0,0,0,1,0,0,0,0,0]],
            ["Hayat Finans", [1,0,0,1,0,0,0,1,1,0]],
            ["Lc Waikiki", [1,1,1,0,1,1,1,0,1,0]],
            ["Yapı Kredi", [0,1,0,1,0,1,0,1,1,0]],
            ["Anadolu Efes", [1,0,1,0,1,1,1,0,0,1]],
            ["Coca Cola", [0,1,0,1,1,1,1,1,1,1]],
            ["Trendyol", [0,1,1,1,1,0,0,1,1,1]]
        ]
    
    def duplicate_data(self, data):
        """16 profili duplike ederek 32 profil oluştur"""
        duplicated_data = np.vstack([data, data])
        return duplicated_data
    
    def fit_regression(self, X, y):
        """Regresyon modelini fit et"""
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        n = len(y)
        k = X.shape[1]
        
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)
        
        if r2 < 1.0:
            msr = r2 * np.var(y) * (n - 1) / (k - 1)
            mse_resid = (1 - r2) * np.var(y) * (n - 1) / (n - k)
            f_stat = msr / mse_resid if mse_resid > 0 else 0
        else:
            f_stat = float('inf')
        
        coefficients = np.concatenate([[model.intercept_], model.coef_])
        
        return {
            'model': model,
            'coefficients': coefficients,
            'r2': r2,
            'adj_r2': adj_r2,
            'rmse': rmse,
            'f_statistic': f_stat,
            'predictions': y_pred
        }
    
    def calculate_career_scores(self, coefficients):
        """Kariyer yolu puanlarını hesapla"""
        scores = {}
        
        for career_name, features in self.career_paths.items():
            score = coefficients[0]  # intercept
            for i, feature_value in enumerate(features):
                score += coefficients[i + 1] * feature_value
            scores[career_name] = score
            
        return scores
    
    def calculate_company_scores(self, coefficients):
        """Şirketlerin puanlarını hesapla"""
        company_scores = []
        
        for company_name, features in self.company_data:
            try:
                # Şirket puanı = intercept + Σ(katsayı × özellik)
                score = coefficients[0]  # intercept
                for i, feature_value in enumerate(features):
                    if i + 1 < len(coefficients):  # Katsayı var mı kontrol et
                        score += coefficients[i + 1] * feature_value
                
                company_scores.append({
                    'name': company_name,
                    'score': score,
                    'features': features
                })
            except Exception as e:
                continue  # Bu şirketi atla
        
        # Puana göre büyükten küçüğe sırala
        company_scores.sort(key=lambda x: x['score'], reverse=True)
        return company_scores
    
    def calculate_company_regrets(self, company_scores, ideal_score):
        """Şirketlerin regret oranlarını hesapla"""
        for company in company_scores:
            if ideal_score != 0:
                regret_percent = ((ideal_score - company['score']) / abs(ideal_score)) * 100
                company['regret_percent'] = max(0, regret_percent)  # Negatif regret durumu için
            else:
                company['regret_percent'] = 0
            company['match_percent'] = 100 - company['regret_percent']
        
        return company_scores
    
    def is_service_sector_career(self, career_name):
        """Kariyer yolunun hizmet sektörü olup olmadığını kontrol et"""
        return career_name in self.service_sector_careers
    
    def analyze_companies(self, coefficients, ideal_score):
        """Şirket analizi yap"""
        # Şirket puanlarını hesapla
        company_scores = self.calculate_company_scores(coefficients)
        
        if not company_scores:
            return None
        
        # Regret oranlarını hesapla
        company_scores = self.calculate_company_regrets(company_scores, ideal_score)
        
        return {
            'all_companies': company_scores,
            'top_5': company_scores[:5],
            'ideal_score': ideal_score,
            'total_companies': len(self.company_data)
        }
    
    def analyze(self, data, person_name="Kullanıcı"):
        """Ana analiz fonksiyonu"""
        # 1. Veriyi duplike et
        duplicated_data = self.duplicate_data(data)
        
        # 2. Veriyi ayır
        X = duplicated_data[:, :-1]
        y = duplicated_data[:, -1]
        
        # 3. Regresyon analizi
        results = self.fit_regression(X, y)
        
        # 4. Kariyer yolu puanları
        career_scores = self.calculate_career_scores(results['coefficients'])
        sorted_careers = sorted(career_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 5. İdeal kariyer yolu
        ideal_profile = []
        for i in range(1, len(results['coefficients'])):
            if results['coefficients'][i] > 0:
                ideal_profile.append(1)
            else:
                ideal_profile.append(0)
        
        ideal_score = results['coefficients'][0]
        for i, value in enumerate(ideal_profile):
            ideal_score += results['coefficients'][i + 1] * value
        
        # 6. En önemli kriterler
        top_factors = sorted(enumerate(results['coefficients'][1:]), 
                           key=lambda x: abs(x[1]), reverse=True)[:5]
        
        # 7. En iyi kariyer yolu
        best_career_name, best_score = sorted_careers[0]
        
        # 8. Şirket analizi (eğer hizmet sektörü önerildiyse)
        company_analysis = None
        if self.is_service_sector_career(best_career_name):
            company_analysis = self.analyze_companies(results['coefficients'], ideal_score)
        
        return {
            'person_name': person_name,
            'sorted_careers': sorted_careers,
            'ideal_score': ideal_score,
            'ideal_profile': ideal_profile,
            'top_factors': top_factors,
            'model_results': results,
            'coefficients': results['coefficients'],
            'best_career': best_career_name,
            'best_score': best_score,
            'company_analysis': company_analysis,
            'is_service_sector': self.is_service_sector_career(best_career_name)
        }

def load_google_sheets_data(sheet_url):
    """Google Sheets'ten veri yükle"""
    try:
        # Google Sheets URL'ini CSV export formatına çevir
        sheet_id = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', sheet_url).group(1)
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        
        # CSV verisini oku
        response = requests.get(csv_url)
        response.raise_for_status()
        
        # pandas ile oku
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        return df
    except Exception as e:
        st.error(f"Google Sheets verisi yüklenemedi: {e}")
        return None

def check_user_data(email, df):
    """Kullanıcının form verilerini kontrol et"""
    if df is None:
        return None, False
    
    try:
        # Email'i C sütununda ara (pandas 0-indexed)
        user_row = df[df.iloc[:, 2].str.lower() == email.lower()]
        
        if user_row.empty:
            return None, False
        
        # M sütunundan AB sütununa kadar profil puanlarını al (M=12, AB=27 -> 12:28)
        profile_scores = user_row.iloc[0, 12:28].values
        
        # NaN değerleri kontrol et
        if pd.isna(profile_scores).any():
            return None, False
        
        return profile_scores, True
    except Exception as e:
        st.error(f"Kullanıcı verisi kontrolünde hata: {e}")
        return None, False

def convert_scores_to_analysis_data(profile_scores):
    """Profil puanlarını analiz formatına çevir"""
    # Sabit profil özellikleri (16 profil x 10 özellik)
    profile_features = [
        [1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
        [0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
        [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 1, 1, 1],
        [0, 1, 0, 1, 1, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 1, 1, 1],
        [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    ]
    
    # Özellikler + puanları birleştir
    analysis_data = []
    for i in range(16):
        row = profile_features[i] + [float(profile_scores[i])]
        analysis_data.append(row)
    
    return np.array(analysis_data)

def create_form_info_page():
    """Form doldurma bilgi sayfası"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b6b, #ffa500); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin: 2rem 0;">
        <h2>📝 Form Doldurma Gerekli</h2>
        <p style="font-size: 1.2rem;">Kariyer analizinizi görebilmek için önce formu doldurmanız gerekiyor.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("### 📋 Form Doldurma Adımları:")
    st.write("""
    1. **Google Forms Linki**: Size verilen Google Forms linkine gidin
    2. **E-posta Adresinizi Girin**: Formda aynı e-posta adresini kullanın
    3. **Profilleri Değerlendirin**: 16 farklı kariyer profilini değerlendirin
    4. **Formu Gönderin**: Tüm soruları yanıtladıktan sonra formu gönderin
    5. **Sonuçları Görün**: Bu sayfaya geri dönüp e-posta adresinizle giriş yapın
    """)
    
    st.info("💡 **İpucu**: Formu doldurduktan sonra bu sayfayı yenileyin ve e-posta adresinizi tekrar girin.")
    
    if st.button("🔄 Sayfayı Yenile"):
        st.rerun()

def create_quick_demo_data():
    """Hızlı demo için örnek veri oluştur"""
    np.random.seed(42)
    data = []
    
    for i in range(16):
        row = []
        # Random binary values for 10 criteria
        for j in range(10):
            row.append(np.random.choice([0, 1]))
        # Random score between 1-9
        row.append(np.random.randint(1, 10))
        data.append(row)
    
    return np.array(data)

def display_company_recommendations(company_analysis):
    """Şirket önerilerini göster"""
    if not company_analysis or not company_analysis['top_5']:
        return
    
    st.markdown("""
    <style>
    .service-sector-alert h2 {
      color: #537F9D !important;
    }
    .service-sector-alert p {
      color: #555555 !important;
    }
    </style>

    <div class="service-sector-alert">
        <h2>🏢 HİZMET SEKTÖRÜ KARİYERİNİZE BAŞLAMANIZ İÇİN GÜZEL BİR NOKTA!</h2>
        <p style="font-size: 1.1rem; margin-bottom: 0;">Size özel şirket önerileri hazırlandı</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("🏆 En Uygun 5 Şirket Önerisi")
    
    # Top 5 companies
    medals = ["🥇", "🥈", "🥉", "🎯", "🎯"]
    
    for i, company in enumerate(company_analysis['top_5']):
        # Regret hesaplama güvenliği
        regret = company.get('regret_percent', 0)
        match = company.get('match_percent', 100)
        
        # Özel durum: İdealden daha iyi şirket
        if regret < 0:
            status_text = "İdealden daha iyi! 🎉"
            status_color = "#27ae60"
        else:
            status_text = f"Pişmanlık: %{regret:.1f}"
            status_color = "#e74c3c" if regret > 15 else "#f39c12" if regret > 5 else "#27ae60"
        
        st.markdown(f"""
        <div class="company-card">
            <h3>{medals[i]} {company['name']}</h3>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p><strong>📊 Puan:</strong> {company['score']:.2f}</p>
                    <p><strong>🎯 Pişmanlık:</strong> %{regret:.1f}</p>
                </div>
                <div style="text-align: right;">
                    <p style="renk: {status_color}; font-weight: bold;">{status_text}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🎯 İdeal Şirket Puanı", 
            f"{company_analysis['ideal_score']:.2f}",
            help="Size en uygun olacak teorik şirket puanı"
        )
    
    with col2:
        st.metric(
            "🏆 En Uygun Şirket", 
            f"{company_analysis['top_5'][0]['score']:.2f}",
            delta=f"{company_analysis['top_5'][0]['score'] - company_analysis['ideal_score']:.2f}",
            help=f"En yüksek puanlı şirket: {company_analysis['top_5'][0]['name']}"
        )
    
    with col3:
        st.metric(
            "📊 Analiz Edilen Şirket", 
            company_analysis['total_companies'],
            help="Veritabanında bulunan toplam şirket sayısı"
        )
    
    # Company scores visualization
    st.subheader("📈 Şirket Puanları Karşılaştırması")
    
    company_names = [comp['name'] for comp in company_analysis['top_5']]
    company_scores = [comp['score'] for comp in company_analysis['top_5']]
    regret_values = [comp.get('regret_percent', 0) for comp in company_analysis['top_5']]
    
    # İki grafik yan yana
    col1, col2 = st.columns(2)
    
    with col1:
        fig_scores = px.bar(
            x=company_scores,
            y=company_names,
            orientation='h',
            title="En Uygun 5 Şirket Puanları",
            labels={'x': 'Puan', 'y': 'Şirket'},
            color=company_scores,
            color_continuous_scale='viridis'
        )
        fig_scores.add_vline(
            x=company_analysis['ideal_score'], 
            line_dash="dash", 
            line_color="red",
            annotation_text="İdeal Puan"
        )
        fig_scores.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_scores, use_container_width=True)
    
    with col2:
        fig_regret = px.bar(
            x=regret_values,
            y=company_names,
            orientation='h',
            title="Pişmanlık Oranları (%)",
            labels={'x': 'Pişmanlık (%)', 'y': 'Şirket'},
            color=regret_values,
            color_continuous_scale='RdYlGn_r'
        )
        fig_regret.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_regret, use_container_width=True)

def display_results(analysis_results):
    """Sonuçları görsel olarak göster"""
    results = analysis_results
    
 
    # Top 3 career recommendations
    st.header("🏆 En Uygun 3 Kariyer Önerisi")
    
    top_3_careers = results['sorted_careers'][:3]
    medals = ["🥇", "🥈", "🥉"]
    colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
    
    for i, (career_name, score) in enumerate(top_3_careers):
        regret = ((results['ideal_score'] - score) / results['ideal_score']) * 100
        match_percentage = (1 - regret/100) * 100
        
        st.markdown(f"""
        <div class="career-card">
            <h3>{medals[i]} {career_name}</h3>
            <p><strong>Puan:</strong> {score:.2f}</p>
            <p><strong>Pişmanlık:</strong> %{regret:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
  
    # Service sector check and company recommendations
    if results['is_service_sector'] and results['company_analysis']:
        st.success(
            "🚀 Önerilen kariyer yolunuz hizmet sektöründe çalışmak. Bu bağlamda aşağıdaki şirketler size uygun olabilir:"
        )
        display_company_recommendations(results['company_analysis'])
        st.markdown("---")
    elif results['is_service_sector']:
        st.info("🏢 Şirket önerileri hazırlanıyor...")
  
    # Career scores visualization
    st.header("📈 Tüm Kariyer Yolları Karşılaştırması")
    
    career_names = [career[0] for career in results['sorted_careers']]
    career_scores = [career[1] for career in results['sorted_careers']]
    
    fig = px.bar(
        x=career_scores,
        y=career_names,
        orientation='h',
        title="Kariyer Yolu Puanları",
        labels={'x': 'Puan', 'y': 'Kariyer Yolu'},
        color=career_scores,
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top criteria analysis
    st.header("🎯 En Önemli Kriterler")

    variable_names_list = [
        "Stres Düzeyi", "Kendini Geliştirme", "Şirket Kültürü",
        "Yan Haklar", "Şirketin Uluslararası Olması", "Ekip Bağları",
        "Yurtdışı İmkanları", "Maaş", "Şirketin Konumu", "Yükselme Potansiyeli"
    ]

    factor_names = [variable_names_list[idx] for idx, _ in results['top_factors']]

    for i, factor_name in enumerate(factor_names):
        rank_emoji = ["🥇", "🥈", "🥉", "🎯", "🎯"][i]
        st.markdown(f"- {rank_emoji} {factor_name}")
    
    # Model statistics
    st.header("📊 Model İstatistikleri")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Değeri", f"{results['model_results']['r2']:.4f}")
    
    with col2:
        st.metric("Adjusted R²", f"{results['model_results']['adj_r2']:.4f}")
    
    with col3:
        st.metric("RMSE", f"{results['model_results']['rmse']:.4f}")
    
    with col4:
        st.metric("F İstatistiği", f"{results['model_results']['f_statistic']:.2f}")

def main():
    st.title("🎯 Kariyer Yolu Öneri Sistemi")
    st.write("Size en uygun kariyer yolunu bulmak için e-posta adresinizi girin.")
    
    # Google Sheets URL
    SHEETS_URL = "https://docs.google.com/spreadsheets/d/17gldbSpMdcLVZApDUTEKWX0pa01B5Dfc8caF7ZUMRHg/edit?usp=sharing"
    
    # Initialize the analysis class
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = CareerAnalysisApp()
    
    # Şirket verileri kontrolü
    st.success(f"✅ {len(st.session_state.analyzer.company_data)} şirket verisi yüklü - Hizmet sektörü önerileri hazır!")
    
    # Email girişi
    st.markdown("""
    <div style="background: linear-gradient(
    135deg,
    #613cb0 0%,
    #a87de0 50%,
    #d7c6f5 100%
  ); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin: 2rem 0;">
        <h2>📧 E-posta Adresinizi Girin</h2>
        <p>Kariyer analizi sonuçlarınızı görmek için formu doldururken kullandığınız e-posta adresini girin.</p>
    </div>
    """, unsafe_allow_html=True)
    
    email = st.text_input("📧 E-posta Adresiniz:", placeholder="ornek@email.com")
    
    if email and '@' in email:
        if st.button("🔍 Sonuçları Getir", type="primary"):
            with st.spinner("🔄 Google Sheets'ten verileriniz kontrol ediliyor..."):
                # Google Sheets'ten veri yükle
                df = load_google_sheets_data(SHEETS_URL)
                
                if df is not None:
                    # Kullanıcı verisini kontrol et
                    profile_scores, form_completed = check_user_data(email, df)
                    
                    if form_completed:
                        # Veriyi analiz formatına çevir
                        analysis_data = convert_scores_to_analysis_data(profile_scores)
                        
                        # Analizi çalıştır
                        results = st.session_state.analyzer.analyze(analysis_data, email.split('@')[0])
                        
                        # Sonuçları göster
                        st.success("✅ Verileriniz bulundu! Analiz sonuçlarınız:")
                        display_results(results)
                        
                    else:
                        st.warning("⚠️ Bu e-posta adresi için tamamlanmış form bulunamadı.")
                        create_form_info_page()
                else:
                    st.error("❌ Google Sheets verilerine erişilemedi. Lütfen daha sonra tekrar deneyin.")
    
    elif email and '@' not in email:
        st.error("❌ Lütfen geçerli bir e-posta adresi girin.")
    
    # Bilgi bölümü
    if not email:
        st.markdown("""
        ---
        ### 📋 Nasıl Çalışır?
        
        1. **📧 E-posta Girin**: Formu doldururken kullandığınız e-posta adresini girin
        2. **🔍 Kontrol**: Sistem Google Sheets'te verilerinizi arar
        3. **📊 Analiz**: Verileriniz bulunursa otomatik analiz yapılır
        4. **🎯 Sonuç**: Size en uygun kariyer yolları önerilir
        5. **🏢 Şirket Önerileri**: Hizmet sektörü önerildiyse en iyi şirketler gösterilir
        
        ### ❓ Sorun Yaşıyor musunuz?
        - E-posta adresinizi doğru yazdığınızdan emin olun
        - Formu tamamen doldurduğunuzdan emin olun
        - Birkaç dakika bekleyip tekrar deneyin
        """)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info("🎯 9 Kariyer Yolu")
        with col2:
            st.info("📊 10 Değerlendirme Kriteri")
        with col3:
            st.info("🔍 16 Profil Analizi")
        with col4:
            st.info(f"🏢 {len(st.session_state.analyzer.company_data)} Şirket")
    
    # Demo seçeneği
    st.markdown("---")
    st.subheader("🎮 Demo Modu")
    st.write("Sistemi test etmek için örnek verilerle demo analizi yapabilirsiniz:")
    
    if st.button("⚡ Demo Analizi Çalıştır"):
        with st.spinner("Demo analizi yapılıyor..."):
            demo_data = create_quick_demo_data()
            results = st.session_state.analyzer.analyze(demo_data, "Demo Kullanıcı")
            st.success("✅ Demo analizi tamamlandı!")
            display_results(results)
    
    # Şirket listesi gösterimi
    st.markdown("---")
    with st.expander("🏢 Mevcut Şirket Listesi"):
        st.write("**Analiz edilen şirketler:**")
        companies = st.session_state.analyzer.company_data
        
        # 3 sütunlu layout
        cols = st.columns(3)
        for i, (company_name, _) in enumerate(companies):
            with cols[i % 3]:
                st.write(f"• {company_name}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; renk: #666; padding: 1rem;">
        <p>🎯 Kariyer Yolu Öneri Sistemi | AI Destekli Kariyer Analizi</p>
        <p>📊 Verileriniz güvenli ve gizlidir | 🔒 GDPR Uyumlu</p>
        <p>🏢 {company_count} şirket ile şirket önerileri | 📈 Gerçek zamanlı analiz</p>
    </div>
    """.format(company_count=len(st.session_state.analyzer.company_data) if 'analyzer' in st.session_state else 34), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
