import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew

st.set_page_config(
    page_title="EEG Seizure Predictor",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load('best_model.pkl')

model = load_model()

def bandpower(psd, freqs, lo, hi):
    freqs = np.asarray(freqs)
    psd   = np.asarray(psd)
    mask  = np.logical_and(freqs >= lo, freqs <= hi)
    return float(np.trapz(psd[mask], freqs[mask]))

def extract_features(segment, fs=256):
    features = []
    for ch in range(segment.shape[0]):
        s     = np.asarray(segment[ch], dtype=np.float64)
        diff1 = np.diff(s)
        diff2 = np.diff(diff1)
        mob   = float(np.std(diff1)) / (float(np.std(s)) + 1e-8)
        comp  = (float(np.std(diff2)) / (float(np.std(diff1)) + 1e-8)) / (mob + 1e-8)
        freqs, psd = sp_signal.welch(s, fs=fs, nperseg=256)
        features.extend([
            float(np.mean(s)),
            float(np.var(s)),
            float(np.std(s)),
            float(kurtosis(s)),
            float(skew(s)),
            mob,
            comp,
            bandpower(psd, freqs, 0.5, 4.0),
            bandpower(psd, freqs, 4.0, 8.0),
            bandpower(psd, freqs, 8.0, 13.0),
            bandpower(psd, freqs, 13.0, 30.0),
        ])
    return np.array(features, dtype=np.float64)

st.title("Patient-Specific Seizure Prediction")
st.markdown("**Advanced Machine Learning · CHB-MIT EEG Dataset · Random Forest + GCN**")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input")
    mode = st.radio("Select input mode", ["Use demo sample", "Upload .npy file"])

    if mode == "Use demo sample":
        st.info("Using a synthetic EEG window for demonstration.")
        np.random.seed(42)
        sample = np.random.randn(18, 2048) * 50
        t = np.linspace(0, 8, 2048)
        for ch in range(6):
            sample[ch] += 80 * np.sin(2 * np.pi * 3 * t)
    else:
        uploaded = st.file_uploader("Upload .npy file — shape (18, 2048)", type=['npy'])
        if uploaded:
            sample = np.load(uploaded)
        else:
            st.warning("Please upload a file or switch to demo mode.")
            st.stop()

    if st.button("▶  Run Prediction", type="primary", use_container_width=True):
        with st.spinner("Extracting features and predicting..."):
            feats = extract_features(sample).reshape(1, -1)
            pred  = model.predict(feats)[0]
            prob  = model.predict_proba(feats)[0]

        st.divider()
        if pred == 1:
            st.error("⚠️  SEIZURE RISK DETECTED")
        else:
            st.success("✅  No Seizure Detected")

        st.metric("Seizure Probability", f"{prob[1]*100:.1f}%")
        st.metric("Normal Probability",  f"{prob[0]*100:.1f}%")

        fig_g, ax_g = plt.subplots(figsize=(4, 0.6))
        ax_g.barh([0], [prob[0]], color='#5DB85D', height=0.4)
        ax_g.barh([0], [prob[1]], left=[prob[0]], color='#E85D5D', height=0.4)
        ax_g.set_xlim(0, 1)
        ax_g.set_yticks([])
        ax_g.set_xticks([0, 0.5, 1])
        ax_g.set_xticklabels(['0%', '50%', '100%'])
        ax_g.set_title('Normal ◄──────► Seizure Risk', fontsize=8)
        st.pyplot(fig_g, use_container_width=True)
        plt.close()

with col2:
    st.subheader("EEG Signal Visualisation")
    ch_names = ['FP1-F7','F7-T7','T7-P7','P7-O1','P7-O2','FP2-F4',
                'F4-C4','C4-P4','P4-O2','FP1-F3','F3-C3','C3-P3',
                'P3-O1','FZ-CZ','CZ-PZ','P3-O2','FP2-F8','F8-T8']
    t_ax  = np.linspace(0, 8, 2048)
    n_show = st.slider("Channels to display", 2, 18, 6)

    fig_eeg, axes = plt.subplots(n_show, 1, figsize=(10, n_show * 1.2), sharex=True)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t_ax, sample[i], linewidth=0.6, color='#4C9BE8')
        ax.set_ylabel(ch_names[i], fontsize=8, rotation=0, labelpad=55)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].set_xlabel("Time (seconds)")
    fig_eeg.suptitle("EEG Channels", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig_eeg, use_container_width=True)
    plt.close()

    st.divider()
    st.subheader("Frequency Spectrum (Channel 1)")
    freqs_plot, psd_plot = sp_signal.welch(
        np.asarray(sample[0], dtype=np.float64), fs=256, nperseg=256
    )
    fig_psd, ax_psd = plt.subplots(figsize=(10, 3))
    ax_psd.semilogy(freqs_plot[:60], psd_plot[:60], color='#E85D5D', linewidth=1.5)
    for lo, hi, name, col in [
        (0.5, 4,  'Delta', '#4C9BE8'),
        (4,   8,  'Theta', '#5DB85D'),
        (8,  13,  'Alpha', '#F0A500'),
        (13, 30,  'Beta',  '#E85D5D'),
    ]:
        ax_psd.axvspan(lo, hi, alpha=0.12, color=col, label=name)
    ax_psd.set_xlabel('Frequency (Hz)')
    ax_psd.set_ylabel('PSD (log scale)')
    ax_psd.set_title('Power Spectral Density with EEG Bands')
    ax_psd.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_psd, use_container_width=True)
    plt.close()

st.divider()
with st.expander("About this system"):
    st.markdown("""
    **Dataset**: CHB-MIT Scalp EEG Database (MIT-CHB_processed, Kaggle)  
    **Features**: 198 features — time-domain (mean, variance, kurtosis, skewness, Hjorth parameters)
    and frequency-domain (delta/theta/alpha/beta bandpower) across 18 EEG channels  
    **Models trained**: Random Forest, SVM, XGBoost, GCN  
    **Class imbalance**: Handled with SMOTE oversampling  
    **Evaluation**: 5-fold stratified cross-validation  
    **Window size**: 8 seconds at 256 Hz (2048 samples per channel)
    """)
