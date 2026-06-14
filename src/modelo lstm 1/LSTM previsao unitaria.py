import os
import queue
import json
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf

# Importando suas funções manuais de tratamento
from utils import preprocess_audio, stft_manual, mel_filterbank

# Configurações de Áudio
SAMPLE_RATE = 16000
DURATION = 1.2  
TARGET_LENGTH = int(DURATION * SAMPLE_RATE)

# Carregar o modelo CRNN e as etiquetas
MODEL_PATH = "modelo_mic_simulacao1.keras"
LABELS_PATH = "labels_mic_simulacao1.json"

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("Certifique-se de que o modelo (.keras) e o .json de labels estão nesta pasta!")

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH) as f:
    label_map = json.load(f)
inv_map = {v: k for k, v in label_map.items()}


class AudioInspectorApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Inspetor de Pipeline de Áudio (CRNN)")
        self.window.geometry("1000x750")
        
        # Guardará o áudio final tratado para conseguirmos ouvir e analisar
        self.audio_tratado_atual = np.zeros(TARGET_LENGTH)
        
        # --- CONTAINER DE BOTÕES ---
        frame_botoes = tk.Frame(window)
        frame_botoes.pack(pady=15)
        
        self.btn_gravar = tk.Button(
            frame_botoes, text="🎤 GRAVAR E ANALISAR (1.2s)", 
            font=("Arial", 12, "bold"), bg="#4CAF50", fg="white",
            command=self.executar_pipeline
        )
        # CORREÇÃO AQUI: Alterado de px para padx
        self.btn_gravar.pack(side=tk.LEFT, padx=10)
        
        self.btn_ouvir = tk.Button(
            frame_botoes, text="🔊 OUVIR ÁUDIO TRATADO", 
            font=("Arial", 12, "bold"), bg="#2196F3", fg="white",
            state=tk.DISABLED, 
            command=self.reproduzir_audio
        )
        # CORREÇÃO AQUI: Alterado de px para padx
        self.btn_ouvir.pack(side=tk.LEFT, padx=10)
        
        # --- LABELS DE STATUS ---
        self.lbl_status = tk.Label(window, text="Status: Aguardando clique...", font=("Arial", 11), fg="gray")
        self.lbl_status.pack()
        
        self.lbl_resultado = tk.Label(window, text="Classificação: -", font=("Arial", 16, "bold"), fg="blue")
        self.lbl_resultado.pack(pady=10)
        
        # --- CONFIGURAÇÃO DOS PLOTS ---
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 5.5))
        self.fig.tight_layout(pad=4.0)
        
        self.ax1.plot(self.audio_tratado_atual)
        self.ax1.set_title("1. Áudio Final Tratado (Forma de Onda Normalizada)")
        self.ax2.imshow(np.zeros((128, 94)), aspect='auto', origin='lower')
        self.ax2.set_title("2. Espectrograma Mel Alimentado na Rede")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def reproduzir_audio(self):
        try:
            self.lbl_status.config(text="🔊 Tocando áudio processado...", fg="#2196F3")
            self.window.update()
            
            sd.play(self.audio_tratado_atual, SAMPLE_RATE)
            sd.wait() 
            
            self.lbl_status.config(text="✅ Áudio reproduzido com sucesso.", fg="green")
        except Exception as e:
            messagebox.showerror("Erro de Reprodução", f"Não foi possível tocar o áudio: {str(e)}")

    def executar_pipeline(self):
        try:
            self.lbl_status.config(text="🎙️ GRAVANDO... Fale agora!", fg="red")
            self.btn_gravar.config(state=tk.DISABLED, bg="#9E9E9E")
            self.btn_ouvir.config(state=tk.DISABLED, bg="#9E9E9E")
            self.window.update()
            
            gravacao = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait() 
            
            audio_bruto = gravacao.flatten()
            
            self.lbl_status.config(text="⚙️ Processando dados e rodando a CRNN...", fg="orange")
            self.window.update()
            
            self.audio_tratado_atual = preprocess_audio(audio_bruto, TARGET_LENGTH, training=False)
            
            spec = stft_manual(self.audio_tratado_atual, n_fft=512, hop_length=160)
            spec_power = np.abs(spec) ** 2
            
            mel_fb = mel_filterbank(SAMPLE_RATE, 512, 128)
            mel_spec = np.dot(mel_fb, spec_power.T)
            
            mel_db = 10 * np.log10(mel_spec + 1e-10)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            
            if mel_db.shape[1] < 94:
                mel_db = np.pad(mel_db, ((0,0),(0,94 - mel_db.shape[1])))
            else:
                mel_db = mel_db[:, :94]
                
            mel_input = np.expand_dims(mel_db, axis=(0, -1)) 
            probs = model.predict(mel_input, verbose=0)[0]
            
            classe_idx = np.argmax(probs)
            classe_nome = inv_map[classe_idx]
            confianca = probs[classe_idx]
            
            self.ax1.clear()
            self.ax1.plot(self.audio_tratado_atual, color='#1f77b4')
            self.ax1.set_title("1. Áudio Final Tratado (Forma de Onda Normalizada)")
            self.ax1.set_xlabel("Amostras")
            self.ax1.set_ylabel("Amplitude")
            self.ax1.grid(True)
            
            self.ax2.clear()
            self.ax2.imshow(mel_db, aspect='auto', origin='lower', cmap='viridis')
            self.ax2.set_title(f"2. Espectrograma Mel Alimentado na Rede (Shape: {mel_db.shape})")
            self.ax2.set_xlabel("Passos de Tempo (Sequência lida pela LSTM)")
            self.ax2.set_ylabel("Frequências (Características da CNN)")
            
            self.lbl_resultado.config(
                text=f"Classificação: {classe_nome.upper()} ({confianca*100:.2f}%)",
                fg="#00AA00" if confianca > 0.72 else "#FF5722"
            )
            self.lbl_status.config(text="✅ Concluído! Pronto para outra gravação.", fg="green")
            self.btn_ouvir.config(state=tk.NORMAL, bg="#2196F3") 
            
        except Exception as e:
            messagebox.showerror("Erro no Pipeline", f"Ocorreu um erro: {str(e)}")
            self.lbl_status.config(text="❌ Falha na execução.", fg="red")
            
        finally:
            self.btn_gravar.config(state=tk.NORMAL, bg="#4CAF50")
            self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioInspectorApp(root)
    root.mainloop()