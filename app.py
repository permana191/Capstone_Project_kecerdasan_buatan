import os
import time
import json
import h5py
import cv2
import gdown
import numpy as np
import tensorflow as tf
from datetime import datetime
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.xception import preprocess_input as preprocess_xception

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# ==========================================
# 1. FUNGSI BEDAH FILE (BYPASS BUG KERAS 3)
# ==========================================
def bersihkan_bug_keras(filepath):
    """
    Menghapus 'quantization_config' yang disuntikkan secara tidak sengaja oleh Keras 3
    saat menyimpan model ke format .h5 di Google Colab.
    """
    try:
        with h5py.File(filepath, 'r+') as f:
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                
                config = json.loads(config_str)
                
                def hapus_kuman(obj):
                    if isinstance(obj, dict):
                        obj.pop('quantization_config', None)
                        for v in obj.values():
                            hapus_kuman(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            hapus_kuman(item)
                
                hapus_kuman(config)
                f.attrs['model_config'] = json.dumps(config).encode('utf-8')
                print(f"[SISTEM] File {filepath} berhasil dibedah dan dibersihkan!")
    except Exception as e:
        print(f"[SISTEM] Gagal membedah {filepath}: {e}")

# ==========================================
# 2. PROSES UNDUH CERDAS, BERSIHKAN & MUAT MODEL
# ==========================================
print("[SISTEM] Menginisialisasi Engine AI (ResNet50 & Xception)...")

RESNET_ID = '194ISRkiUYhBoRHK288wY-pcK3GIghLbP'
XCEPTION_ID = '1o2hyJI6P8QTcwYPK3ag5n_XmLKjGC7g9'
resnet_path = 'models/resnet50_best.h5'
xception_path = 'models/xception_best.h5'

def cek_dan_unduh(filepath, gdrive_id, nama_model):
    """
    Mengecek apakah file ada dan ukurannya valid. 
    Jika file korup/terlalu kecil (< 10MB), hapus dan unduh ulang.
    """
    if os.path.exists(filepath):
        ukuran_mb = os.path.getsize(filepath) / (1024 * 1024)
        if ukuran_mb < 10.0:  # Model AI pasti lebih besar dari 10MB
            print(f"[WARNING] File {nama_model} korup atau terlalu kecil ({ukuran_mb:.2f} MB). Menghapus file...")
            os.remove(filepath)
        else:
            print(f"[SISTEM] File {nama_model} aman dan siap digunakan ({ukuran_mb:.2f} MB).")
            return

    print(f"[DOWNLOAD] Mengunduh {nama_model} dari Google Drive...")
    gdown.download(f'https://drive.google.com/uc?id={gdrive_id}', filepath, quiet=False)

# Eksekusi fungsi download cerdas
cek_dan_unduh(resnet_path, RESNET_ID, "ResNet50")
cek_dan_unduh(xception_path, XCEPTION_ID, "Xception")

# Bersihkan file sebelum di-load oleh TensorFlow
print("[SISTEM] Memulai pembersihan file .h5...")
bersihkan_bug_keras(resnet_path)
bersihkan_bug_keras(xception_path)

# Load model ke memori
MODEL_RESNET = load_model(resnet_path)
MODEL_XCEPTION = load_model(xception_path)
print("[SISTEM] Seluruh Engine AI Siap Digunakan!")

# ==========================================
# 3. KONFIGURASI GLOBAL & HELPER FUNCTIONS
# ==========================================
CLASS_NAMES = ['Produk Cacat (Defect)', 'Produk Normal (OK)']
THRESHOLD_LIMIT = 85.0 # Ambang batas Human-in-the-Loop (85%)

# Memori Global untuk Logs & Data Grafik (Chart.js)
inspection_logs = []
dashboard_stats = {
    'total': 0, 'normal': 0, 'defect': 0, 'review': 0, 
    'resnet_times': [], 'xcep_times': []
}

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    return None

def generate_gradcam(img_path, model, preprocess_func, filename_prefix):
    try:
        img = image.load_img(img_path, target_size=(300, 300))
        img_array = image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_func(img_array_expanded)

        last_conv_layer_name = get_last_conv_layer(model)
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_preprocessed)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        original_img = cv2.imread(img_path)
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_TURBO)
        
        superimposed_img = heatmap_colored * 0.5 + original_img
        heatmap_filename = f"hm_{filename_prefix}_{os.path.basename(img_path)}"
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
        cv2.imwrite(heatmap_path, superimposed_img)
        
        return heatmap_filename
    except Exception as e:
        return os.path.basename(img_path)

def run_inference(filepath, filename, model_name, use_gradcam=True):
    global dashboard_stats
    img = image.load_img(filepath, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    start_time = time.time()
    if model_name == 'resnet50':
        x = preprocess_resnet(x)
        preds = MODEL_RESNET.predict(x, verbose=0)
        hm_name = generate_gradcam(filepath, MODEL_RESNET, preprocess_resnet, "res") if use_gradcam else None
        inf_time = time.time() - start_time
        dashboard_stats['resnet_times'].append(inf_time)
    else:
        x = preprocess_xception(x)
        preds = MODEL_XCEPTION.predict(x, verbose=0)
        hm_name = generate_gradcam(filepath, MODEL_XCEPTION, preprocess_xception, "xcep") if use_gradcam else None
        inf_time = time.time() - start_time
        dashboard_stats['xcep_times'].append(inf_time)
        
    raw_class = CLASS_NAMES[np.argmax(preds, axis=1)[0]]
    confidence = float(np.max(preds) * 100)
    
    # LOGIKA HUMAN-IN-THE-LOOP
    if confidence < THRESHOLD_LIMIT:
        final_class = "⚠️ Inspeksi Manual"
        dashboard_stats['review'] += 1
    else:
        final_class = raw_class
        if 'Normal' in final_class:
            dashboard_stats['normal'] += 1
        else:
            dashboard_stats['defect'] += 1
            
    dashboard_stats['total'] += 1
        
    return final_class, confidence, inf_time, hm_name

# ==========================================
# 4. ROUTING & LOGIKA UTAMA APLIKASI
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def index():
    global inspection_logs, dashboard_stats
    
    avg_res = sum(dashboard_stats['resnet_times']) / len(dashboard_stats['resnet_times']) if dashboard_stats['resnet_times'] else 0
    avg_xcep = sum(dashboard_stats['xcep_times']) / len(dashboard_stats['xcep_times']) if dashboard_stats['xcep_times'] else 0
    stats_data = {
        'normal': dashboard_stats['normal'], 'defect': dashboard_stats['defect'], 'review': dashboard_stats['review'],
        'avg_res': round(avg_res, 3), 'avg_xcep': round(avg_xcep, 3)
    }

    if request.method == 'POST':
        files = request.files.getlist('file')
        model_choice = request.form.get('model_choice')
        timestamp = datetime.now().strftime("%H:%M:%S")

        if not files or files[0].filename == '':
            return render_template('index.html', logs=inspection_logs, stats=stats_data)

        # BATCH MODE
        if len(files) > 1:
            batch_results = []
            active_model = 'xception' if model_choice == 'compare' else model_choice 
            
            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    p_class, p_conf, p_time, _ = run_inference(filepath, filename, active_model, use_gradcam=False)
                    
                    batch_results.append({
                        'file': filename, 'status': p_class, 
                        'conf': round(p_conf, 2), 'time': round(p_time, 3)
                    })
                    
                    inspection_logs.insert(0, {'time': timestamp, 'model': active_model.capitalize(), 'status': p_class.split()[1] if len(p_class.split())>1 else p_class, 'speed': f"{p_time:.3f}s"})
            
            stats_data['normal'], stats_data['defect'], stats_data['review'] = dashboard_stats['normal'], dashboard_stats['defect'], dashboard_stats['review']
            
            return render_template('index.html', mode='batch', batch_results=batch_results, 
                                   model_used=active_model.capitalize(), logs=inspection_logs[:15], stats=stats_data)

        # SINGLE / COMPARE MODE
        else:
            file = files[0]
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if model_choice == 'compare':
                res_class, res_conf, res_time, res_hm = run_inference(filepath, filename, 'resnet50', use_gradcam=True)
                xcep_class, xcep_conf, xcep_time, xcep_hm = run_inference(filepath, filename, 'xception', use_gradcam=True)
                
                stats_data['normal'], stats_data['defect'], stats_data['review'] = dashboard_stats['normal'], dashboard_stats['defect'], dashboard_stats['review']
                inspection_logs.insert(0, {
                    'time': timestamp, 'model': 'Head-to-Head',
                    'status': f"R:{res_class.split()[1] if len(res_class.split())>1 else res_class} | X:{xcep_class.split()[1] if len(xcep_class.split())>1 else xcep_class}",
                    'speed': f"R:{res_time:.2f}s | X:{xcep_time:.2f}s"
                })
                
                return render_template('index.html', mode='compare', 
                                       res_data={'class': res_class, 'conf': round(res_conf, 2), 'time': round(res_time, 3), 'hm': url_for('static', filename=f'uploads/{res_hm}')},
                                       xcep_data={'class': xcep_class, 'conf': round(xcep_conf, 2), 'time': round(xcep_time, 3), 'hm': url_for('static', filename=f'uploads/{xcep_hm}')},
                                       logs=inspection_logs[:15], stats=stats_data)
            else:
                p_class, p_conf, p_time, hm = run_inference(filepath, filename, model_choice, use_gradcam=True)
                model_disp = "Xception" if model_choice == 'xception' else "ResNet50"
                
                stats_data['normal'], stats_data['defect'], stats_data['review'] = dashboard_stats['normal'], dashboard_stats['defect'], dashboard_stats['review']
                inspection_logs.insert(0, {'time': timestamp, 'model': model_disp, 'status': p_class.split()[1] if len(p_class.split())>1 else p_class, 'speed': f"{p_time:.3f}s"})
                
                return render_template('index.html', mode='single', 
                                       result=p_class, confidence=round(p_conf, 2), time=round(p_time, 3),
                                       model_used=model_disp, hm_url=url_for('static', filename=f'uploads/{hm}'),
                                       logs=inspection_logs[:15], stats=stats_data)
                                   
    return render_template('index.html', logs=inspection_logs[:15], stats=stats_data)

if __name__ == '__main__':
    # Menggunakan port environment variable agar kompatibel dengan Railway & Gunicorn
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
