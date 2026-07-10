import os
import gdown
import shutil
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ==============================================================================
# 1. CONFIGURATION DE LA PAGE (Mode Dashboard Large)
# ==============================================================================
st.set_page_config(
    page_title="Vision Core IA | SaaS Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. INJECTION DU THÈME CSS "SAAS" (Design clair, cartes, ombres)
# ==============================================================================
st.markdown("""
<style>
    /* Fond général de l'application gris très clair */
    .stApp {
        background-color: #F4F7FC;
    }
    /* Style de la barre latérale blanche */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    /* Création de l'effet "Carte blanche" pour les graphiques */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    /* Cacher le menu hamburger par défaut de Streamlit pour faire plus pro */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 3. LOGIQUE AUTOMATISÉE DE L'IA (Ton code d'origine)
# ==============================================================================
FILE_ID = "1PZdv-iZB6bA-cGkg5wGZlb4qQmja60un"
url = f"https://drive.google.com/uc?id={FILE_ID}"
LOCAL_TEMP_MODEL = "cerveau_weights_model.ckpt"

# Téléchargement initial si le fichier est absent
if not os.path.exists(LOCAL_TEMP_MODEL):
    with st.spinner("🧠 Initialisation du système : Téléchargement du modèle d'IA (Patientez...)"):
        gdown.download(url, LOCAL_TEMP_MODEL, quiet=False)

def verifier_et_placer_modele(nom_materiel):
    """Crée le sous-dossier requis à la volée s'il n'existe pas et y copie le modèle."""
    target_dir = f"./patchcore_results/Patchcore/MVTecAD/{nom_materiel}/v0/weights"
    target_file = os.path.join(target_dir, "model.ckpt")
    
    if not os.path.exists(target_file):
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(LOCAL_TEMP_MODEL, target_file)


# ==============================================================================
# 4. BARRE LATÉRALE (SIDEBAR - MENU SAAS)
# ==============================================================================
with st.sidebar:
    st.title("🔬 Vision Core IA")
    st.markdown("---")
    
    # Menu de navigation factice pour le look SaaS
    menu_choisi = st.radio("Navigation", ["🏠 Dashboard", "📂 Inspection", "📊 Rapports", "⚙️ Paramètres"])
    
    st.markdown("---")
    # C'EST ICI QUE SE TROUVE LE VRAI SÉLECTEUR DE MATÉRIEL QUI ACTIVE TON CODE
    st.subheader("Configuration Système")
    materiel_selectionne = st.selectbox(
        "Matériel à inspecter :", 
        ["Écrou en métal (metal_nut)", "Transistor (transistor)", "Textile (carpet)"]
    )
    
    # Activation de la logique de déplacement de fichier selon le choix
    if "metal_nut" in materiel_selectionne:
        verifier_et_placer_modele("metal_nut")
        id_materiel = "metal_nut"
    elif "transistor" in materiel_selectionne:
        verifier_et_placer_modele("transistor")
        id_materiel = "transistor"
    else:
        verifier_et_placer_modele("carpet")
        id_materiel = "carpet"

    st.success(f"✅ Modèle prêt pour : {id_materiel}")


# ==============================================================================
# 5. CONTENU PRINCIPAL (LE DASHBOARD INSPIRÉ DE TON IMAGE)
# ==============================================================================
st.title("Overview & Metrics")
st.markdown("Surveillance de l'état du système d'inspection par vision artificielle.")
st.write("") # Espacement

# Création de deux colonnes pour la première rangée
col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.subheader("Key Metrics (Inspections)")
        # Une métrique avec une flèche verte de progression
        st.metric(label="Total Pièces Analysées", value="8,200", delta="5.8% (ce mois-ci)")
        
        # Graphique en ligne (Simulé pour ressembler à l'image)
        chart_data_line = pd.DataFrame(np.random.randn(20, 1).cumsum(), columns=['Inspections réussies'])
        st.line_chart(chart_data_line, height=200)

with col2:
    with st.container():
        st.subheader("Weekly Sales / Anomalies Detectées")
        
        # Graphique en barres (Simulé pour ressembler à l'image)
        chart_data_bar = pd.DataFrame({
            'Jours': ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'],
            'Valeurs': [30, 45, 60, 70, 65, 90, 80]
        }).set_index('Jours')
        st.bar_chart(chart_data_bar, height=250)

st.write("") # Espacement

# Création de deux colonnes pour la deuxième rangée
col3, col4 = st.columns(2)

with col3:
    with st.container():
        st.subheader("Task Overview")
        # Création d'un Donut Chart avec Altair (qui est natif à Streamlit) pour reproduire le camembert
        source = pd.DataFrame({"Statut": ['Completed', 'In Progress', 'Pending'], "Valeur": [50, 30, 20]})
        base = alt.Chart(source).encode(
            theta=alt.Theta("Valeur:Q", stack=True), 
            color=alt.Color("Statut:N", scale=alt.Scale(range=['#10B981', '#3B82F6', '#9CA3AF']))
        )
        pie = base.mark_arc(innerRadius=60)
        st.altair_chart(pie, use_container_width=True)

with col4:
    with st.container():
        st.subheader("Recent Activities")
        # Création d'un tableau propre pour l'historique
        df_activities = pd.DataFrame({
            "Activity": ["Connexion utilisateur", "Changement de modèle", "Inspection (Écrou)", "Génération Rapport"],
            "Date": ["Aujourd'hui", "Aujourd'hui", "Hier", "Hier"],
            "Status": ["Succès", "Succès", "Échec", "Succès"]
        })
        # Affichage sans l'index pour faire plus propre
        st.dataframe(df_activities, hide_index=True, use_container_width=True)

# Ici, tu pourras rajouter en dessous la zone d'upload de tes images pour tester l'IA !
st.markdown("---")
st.subheader(f"🛠️ Interface d'Inspection : {materiel_selectionne}")
uploaded_file = st.file_uploader("Chargez une image pour analyse avec le modèle actif", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    st.info("L'image est chargée et prête à être passée dans le réseau PatchCore.")
