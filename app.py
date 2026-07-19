import os
import time
import cv2
import numpy as np
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
import gdown
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

# ==============================================================================
# ALGORITMA SMART DOWNLOADER (GOOGLE DRIVE)
# Mencegah konflik dengan file Pointer LFS yang tertinggal di GitHub
# ==============================================================================
os.makedirs('models', exist_ok=True)

RESNET_ID = '194ISRkiUYhBoRHK288wY-pcK3GIghLbP'
XCEPTION_ID = '1o2hyJI6P8QTcwYPK3ag5n_XmLKjGC7g9'

resnet_path = 'models/resnet50_best.h5'
xception_path = 'models/xception_best.h5'

def download_model_if_needed(file_id, output_path, model_name):
    # Jika file ada, tetapi ukurannya di bawah 10MB (kemungkinan file Pointer LFS palsu), hapus!
    if os.path.exists(output_path) and os.path.getsize(output_path) < 10000000:
        print(f"[SISTEM] Mendeteksi file {model_name} palsu/rusak. Menghapus file lama...")
        os.remove(output_path)
        
    # Jika file benar-benar tidak ada, mulai unduh dari Google Drive
    if not os.path.exists(output_path):
        print(f"[DOWNLOAD] Memulai pengunduhan {model_name} asli dari Google Drive...")
        gdown.download(id=file_id, output=output_path, quiet=False)
        print(f"[DOWNLOAD] {model_name} berhasil diunduh dan diamankan!")

print("[SISTEM] Memvalidasi integritas file model AI...")
download_model_if_needed(RESNET_ID, resnet_path, "ResNet50")
download_model_if_needed(XCEPTION_ID, xception_path, "Xception")

print("[SISTEM] Menginisialisasi Engine AI (ResNet50 & Xception)...")
MODEL_RESNET = load_model(resnet_path)
MODEL_XCEPTION = load_model(xception_path)
print("[SISTEM] Seluruh Engine AI Siap Digunakan!")
# ==============================================================================

CLASS_NAMES = ['Produk Cacat (Defect)', 'Produk Normal (OK)']
THRESHOLD_LIMIT = 85.0 

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

@app.route('/', methods=['GET', 'POST'])
def index():
    global inspection_logs, dashboard_stats
    
    avg_res = sum(dashboard_stats['resnet_times']) / len(dashboard_stats['resnet_times']) if dashboard_stats['resnet_times'] else 0
    avg_xcep = sum(dashboard_stats['xcep_times']) / len(dashboard_stats['xcep_times']) if dashboard_stats['xcep_times'] else 0
    stats_data = {
        'normal': dashboard_stats['normal'], 'defect': dashboard_stats['defect'], 'review': dashboard_stats['review'],
        'avg_res': round(avg_res, 3), 'avg_xcep': round(avg_xcep, 3),
        'total': dashboard_stats['total']
    }

    if request.method == 'POST':
        files = request.files.getlist('file')
        model_choice = request.form.get('model_choice')
        timestamp = datetime.now().strftime("%H:%M:%S")

        if not files or files[0].filename == '':
            return render_template('index.html', logs=inspection_logs, stats=stats_data)

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
                    
            stats_data['normal'], stats_data['defect'], stats_data['review'], stats_data['total'] = dashboard_stats['normal'], dashboard_stats['defect'], dashboard_stats['review'], dashboard_stats['total']
            
            return render_template('index.html', mode='batch', batch_results=batch_results, 
                                   model_used=active_model.capitalize(), stats=stats_data)

        else:
            file = files[0]
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if model_choice == 'compare':
                res_class, res_conf, res_time, res_hm = run_inference(filepath, filename, 'resnet50', use_gradcam=True)
                xcep_class, xcep_conf, xcep_time, xcep_hm = run_inference(filepath, filename, 'xception', use_gradcam=True)
                
                stats_data['normal'], stats_data['defect'], stats_data['review'], stats_data['total'] = dashboard_stats['normal'], dashboard_stats['defect'], dashboard_stats['review'], dashboard_stats['total']
                
                return render_template('index.html', mode='compare', 
                                       res_data={'class': res_class, 'conf': round(res_conf, 2), 'time': round(res_time, 3), 'hm': url_for('static', filename=f'uploads/{res_hm}')},
                                       xcep_data={'class': xcep_class, 'conf': round(xcep_conf, 2), 'time': round(xcep_time, 3), 'hm': url_for('static', filename=f'uploads/{xcep_hm}')},
                                       stats=stats_data)
            else:
                p_class, p_conf, p_time, hm = run_inference(filepath, filename, model_choice, use_gradcam=True)
                model_disp = "Xception" if model_choice == 'xception' else "ResNet50"
                
                stats_data['normal'], stats_data['defect'], stats_data['review'], stats_data['total'] = dashboard_stats['normal'], dashboard_stats['defect'], dashboard_stats['review'], dashboard_stats['total']
                
                return render_template('index.html', mode='single', 
                                       result=p_class, confidence=round(p_conf, 2), time=round(p_time, 3),
                                       model_used=model_disp, hm_url=url_for('static', filename=f'uploads/{hm}'),
                                       stats=stats_data)
                                   
    return render_template('index.html', stats=stats_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
