from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash, send_file
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
from gtts import gTTS
import os
import tempfile
from datetime import datetime
import threading
import time
import subprocess
import json
import traceback

# Tente importar o auth de forma mais segura
try:
    from auth import user_manager, User
    print("✅ Módulo auth carregado com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar auth: {e}")
    # Criar um fallback básico
    class User(UserMixin):
        def __init__(self, id, username, is_admin=False):
            self.id = id
            self.username = username
            self.is_admin = is_admin
        
        def get_id(self):
            return str(self.id)
        
        def is_admin(self):
            return self.is_admin
    
    class UserManager:
        def get_user(self, user_id):
            # Usuário admin básico para teste
            if user_id == "1":
                return User(1, "admin", True)
            return None
        
        def authenticate(self, username, password):
            # Login básico para teste
            if username == "admin" and password == "admin":
                return User(1, "admin", True)
            return None
        
        def get_stats(self):
            return {"total_users": 1, "active_users": 1}
    
    user_manager = UserManager()

app = Flask(__name__)
app.secret_key = 'tradulibras_secret_key_2024'

# Configurar Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Por favor, faça login para acessar esta página.'
login_manager.login_message_category = 'info'

# Variáveis globais para compartilhamento entre threads
camera_frame = None
camera_lock = threading.Lock()
camera_running = False
camera_initialized = False
camera_index = 0

# Variáveis do modelo (serão inicializadas depois)
model = None
model_info = {'classes': []}

# Global variables for text formation
current_letter = ""
formed_text = ""
corrected_text = ""
last_prediction_time = datetime.now()
prediction_cooldown = 2.5  # segundos
letter_detected = False

@login_manager.user_loader
def load_user(user_id):
    try:
        return user_manager.get_user(user_id)
    except Exception as e:
        print(f"❌ Erro no user_loader: {e}")
        return None

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def load_model():
    """Carregar o modelo de forma segura"""
    global model, model_info
    
    # Load the trained model (procurando em múltiplos caminhos e ignorando arquivos vazios)
    try:
        import os
        candidate_model_paths = [
            'modelos/modelo_libras_expandido.pkl',
            'modelo_libras_expandido.pkl',
            'modelos/modelo_libras.pkl',
            'modelo_libras.pkl'
        ]
        selected_model_path = None
        for path in candidate_model_paths:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                selected_model_path = path
                break
        if selected_model_path is None:
            raise FileNotFoundError("Nenhum arquivo de modelo válido encontrado (todos ausentes ou vazios)")

        with open(selected_model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Modelo carregado com sucesso de: {selected_model_path}")
        if hasattr(model, 'classes_'):
            print(f"📊 Classes do modelo: {list(model.classes_)}")
        else:
            print("⚠️  Modelo não tem atributo 'classes_'")
        if hasattr(model, 'n_features_in_'):
            print(f"🔢 Features esperadas pelo modelo: {model.n_features_in_}")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        model = None

    # Load model info (mesma estratégia de múltiplos caminhos)
    try:
        import os
        candidate_info_paths = [
            'modelos/modelo_info_expandido.pkl',
            'modelo_info_expandido.pkl',
            'modelos/modelo_info.pkl',
            'modelo_info.pkl'
        ]
        selected_info_path = None
        for path in candidate_info_paths:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                selected_info_path = path
                break
        if selected_info_path is None:
            raise FileNotFoundError("Nenhum arquivo de info de modelo válido encontrado (ausente ou vazio)")

        with open(selected_info_path, 'rb') as f:
            model_info = pickle.load(f)
        print(f"📋 Informações do modelo carregadas de: {selected_info_path}")
        if 'classes' in model_info:
            print(f"🔤 Classes disponíveis: {model_info['classes']}")
    except Exception as e:
        print(f"❌ Erro ao carregar info do modelo: {e}")
        model_info = {'classes': []}

def find_working_camera():
    """Encontrar uma câmera que funcione"""
    print("🔍 Procurando câmera disponível...")
    for i in range(3):  # Testar índices 0, 1, 2
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Câmera encontrada no índice {i}")
                    cap.release()
                    return i
                cap.release()
        except Exception as e:
            print(f"❌ Erro com câmera {i}: {e}")
    
    print("❌ Nenhuma câmera funcionando encontrada")
    return 0  # Retornar 0 como padrão mesmo que não funcione

# =========================================
# Inicialização do MediaPipe ajustada
# =========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# =========================================
# Função de processamento de landmarks melhorada
# =========================================
def process_landmarks(hand_landmarks):
    """Processar landmarks normalizando pela mão e mantendo 63 features"""
    try:
        if not hand_landmarks or len(hand_landmarks.landmark) != 21:
            return None
        
        # Ponto de referência (pulso)
        p0 = hand_landmarks.landmark[0]
        points = []
        for lm in hand_landmarks.landmark:
            points.extend([lm.x - p0.x, lm.y - p0.y, lm.z - p0.z])
        
        # Normalizar pelo tamanho da mão
        points_np = np.array(points).reshape(-1, 3)
        max_dist = np.max(np.linalg.norm(points_np, axis=1))
        if max_dist > 0:
            points_np /= max_dist
        
        points_flat = points_np.flatten()
        if len(points_flat) == 63:
            return points_flat
        else:
            print(f"⚠️ Número incorreto de pontos após normalização: {len(points_flat)}")
            return None
        
    except Exception as e:
        print(f"❌ Erro no processamento de landmarks: {e}")
        return None

# =========================================
# Loop da câmera ajustado para cooldown confiável
# =========================================
def init_camera():
    global camera_running, camera_initialized, camera_index, last_prediction_time, prediction_cooldown

    if 'last_prediction_time' not in globals():
        last_prediction_time = datetime.now()
    if 'prediction_cooldown' not in globals():
        prediction_cooldown = 1.2  # menor cooldown, detecta letras mais rápido

    def camera_worker():
        global camera, camera_running, camera_frame, current_letter, letter_detected, last_prediction_time

        hands_instance = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        camera = cv2.VideoCapture(camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while camera_running:
            success, frame = camera.read()
            if not success or frame is None:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_instance.process(rgb_frame)

            if results and results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = process_landmarks(hand_landmarks)

                if landmarks is not None and model is not None:
                    now = datetime.now()
                    elapsed = (now - last_prediction_time).total_seconds()
                    if elapsed >= prediction_cooldown:
                        try:
                            landmarks_np = np.array(landmarks).reshape(1, -1)
                            prediction = model.predict(landmarks_np)
                            current_letter = prediction[0]
                            letter_detected = True
                            last_prediction_time = now
                        except Exception as e:
                            print(f"❌ Erro na predição: {e}")
                            current_letter = ""
                            letter_detected = False
                    else:
                        letter_detected = False
                else:
                    current_letter = ""
                    letter_detected = False

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                current_letter = ""
                letter_detected = False

            with camera_lock:
                camera_frame = frame

            time.sleep(0.02)

        camera.release()
        hands_instance.close()

    if not camera_running:
        camera_running = True
        camera_initialized = True
        threading.Thread(target=camera_worker, daemon=True).start()






def generate_frames():
    """Gerar frames para o streaming"""
    global camera_frame, camera_running, camera_initialized
    
    # Aguardar inicialização da câmera
    wait_start = time.time()
    while not camera_initialized and (time.time() - wait_start) < 10:
        time.sleep(0.1)
    
    while camera_running:
        with camera_lock:
            if camera_frame is not None:
                frame = camera_frame.copy()
            else:
                # Frame preto se não houver câmera
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera não disponivel", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Codificar frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Frame de erro
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Erro ao codificar frame", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)

# =============================================================================
# ROTAS DA APLICAÇÃO
# =============================================================================

@app.route('/')
def index():
    try:
        if current_user.is_authenticated:
            if hasattr(current_user, 'is_admin') and current_user.is_admin():
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('camera'))
        return redirect(url_for('login'))
    except Exception as e:
        print(f"❌ Erro na rota /: {e}")
        traceback.print_exc()
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            
            if not username or not password:
                flash('Por favor, preencha todos os campos!', 'error')
                return render_template('login.html')
            
            user = user_manager.authenticate(username, password)
            if user:
                login_user(user)
                flash(f'Bem-vindo, {user.username}!', 'success')
                
                if hasattr(user, 'is_admin') and user.is_admin():
                    return redirect(url_for('admin_dashboard'))
                else:
                    return redirect(url_for('camera'))
            else:
                flash('Usuário ou senha incorretos!', 'error')
        
        return render_template('login.html')
    except Exception as e:
        print(f"❌ Erro na rota /login: {e}")
        traceback.print_exc()
        flash('Erro interno no servidor. Tente novamente.', 'error')
        return render_template('login.html')

@app.route('/admin')
@login_required
def admin_dashboard():
    try:
        if not (hasattr(current_user, 'is_admin') and current_user.is_admin()):
            flash('Acesso negado! Apenas administradores podem acessar esta página.', 'error')
            return redirect(url_for('camera'))
        
        user_stats = user_manager.get_stats()
        return render_template('admin_dashboard.html', user_stats=user_stats)
    except Exception as e:
        print(f"❌ Erro na rota /admin: {e}")
        traceback.print_exc()
        flash('Erro ao carregar o dashboard.', 'error')
        return redirect(url_for('camera'))

@app.route('/camera')
@login_required
def camera():
    try:
        return render_template('camera_tradulibras.html')
    except Exception as e:
        print(f"❌ Erro na rota /camera: {e}")
        traceback.print_exc()
        return "Erro ao carregar a câmera", 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    global formed_text, corrected_text, current_letter
    return jsonify({
        'current_letter': current_letter,
        'formed_text': formed_text,
        'corrected_text': corrected_text
    })

@app.route('/clear_text', methods=['POST'])
def clear_text():
    global formed_text, corrected_text, current_letter, letter_detected
    formed_text = ""
    corrected_text = ""
    current_letter = ""
    letter_detected = False
    return jsonify({
        'status': 'success',
        'message': 'Texto limpo com sucesso'
    })

@app.route('/letra_atual')
def letra_atual():
    global current_letter, letter_detected
    letra_para_retornar = current_letter if current_letter and current_letter.strip() else "-"
    
    return jsonify({
        'letra': letra_para_retornar,
        'detectada': letter_detected
    })

@app.route('/falar_texto', methods=['POST'])
def falar_texto():
    global corrected_text, formed_text
    
    data = request.get_json()
    texto = data.get('texto', '') if data else ''
    
    if not texto:
        texto = corrected_text if corrected_text else formed_text
    
    if not texto or texto.strip() == "":
        return jsonify({'error': 'Nenhum texto para falar'})
    
    try:
        tts = gTTS(text=texto, lang='pt-br', slow=False)
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time())
        temp_file = os.path.join(temp_dir, f'speech_{timestamp}.mp3')
        tts.save(temp_file)
        
        return send_file(temp_file, mimetype='audio/mpeg', as_attachment=False)
    except Exception as e:
        return jsonify({'error': f'Erro na síntese de voz: {str(e)}'})

@app.route('/status')
def status():
    """Rota para verificar o status da aplicação"""
    try:
        model_loaded = model is not None
        model_classes = []
        model_features = 0
        
        if model_loaded:
            if hasattr(model, 'classes_'):
                model_classes = list(model.classes_)
            if hasattr(model, 'n_features_in_'):
                model_features = model.n_features_in_
        
        # Testar predição simples
        test_prediction = None
        if model_loaded and model_features > 0:
            try:
                dummy_data = np.zeros((1, model_features))
                test_prediction = model.predict(dummy_data)[0]
            except Exception as e:
                test_prediction = f"Erro: {e}"
        
        return jsonify({
            'status': 'online',
            'model_loaded': model_loaded,
            'model_classes': model_classes,
            'model_features': model_features,
            'test_prediction': test_prediction,
            'model_info_classes': model_info.get('classes', []),
            'camera_available': camera_running,
            'camera_initialized': camera_initialized,
            'current_letter': current_letter,
            'formed_text': formed_text,
            'prediction_cooldown': prediction_cooldown
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/test_prediction', methods=['GET'])
def test_prediction():
    """Testar se o modelo está funcionando"""
    if model is None:
        return jsonify({'error': 'Modelo não carregado'})
    
    try:
        # Criar dados de teste simples
        if hasattr(model, 'n_features_in_'):
            features = model.n_features_in_
        else:
            features = 63  # Valor padrão
            
        test_data = np.zeros((1, features))
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(test_data)
            confidence = np.max(probabilities)
            predicted_class_idx = np.argmax(probabilities)
            
            if hasattr(model, 'classes_'):
                predicted_letter = model.classes_[predicted_class_idx]
            else:
                predicted_letter = "Classe " + str(predicted_class_idx)
        else:
            prediction = model.predict(test_data)
            predicted_letter = prediction[0]
            confidence = 0.5
        
        return jsonify({
            'prediction': predicted_letter,
            'confidence': float(confidence),
            'features_used': features,
            'model_has_classes': hasattr(model, 'classes_'),
            'model_has_proba': hasattr(model, 'predict_proba')
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/restart_camera', methods=['POST'])
@login_required
def restart_camera():
    """Reiniciar a câmera"""
    global camera_running, camera_initialized
    
    camera_running = False
    time.sleep(2)
    init_camera()
    
    return jsonify({'status': 'success', 'message': 'Câmera reiniciada'})

@app.route('/debug')
def debug():
    """Página de debug completa"""
    return jsonify({
        'camera_running': camera_running,
        'camera_initialized': camera_initialized,
        'camera_frame_available': camera_frame is not None,
        'camera_index': camera_index,
        'model_loaded': model is not None,
        'model_classes': model_info.get('classes', []),
        'current_letter': current_letter,
        'formed_text': formed_text,
        'corrected_text': corrected_text,
        'mediapipe_initialized': hands is not None
    })

# =============================================================================
# INICIALIZAÇÃO
# =============================================================================

if __name__ == '__main__':
    import socket
    
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    local_ip = get_local_ip()
    
    print("🚀 Iniciando TraduLibras v2.0.0...")
    print("=" * 50)
    
    # Verificar se os templates existem
    template_files = ['login.html', 'camera_tradulibras.html', 'admin_dashboard.html']
    template_dir = 'templates'
    
    if os.path.exists(template_dir):
        print("📁 Templates encontrados:")
        for file in os.listdir(template_dir):
            print(f"   ✅ {file}")
    else:
        print("❌ Pasta templates não encontrada!")
    
    print("=" * 50)
    
    # CARREGAR O MODELO PRIMEIRO
    print("🤖 Carregando modelo...")
    load_model()
    
    print("📊 Informações do sistema:")
    print(f"   Modelo carregado: {model is not None}")
    
    if model is not None:
        if hasattr(model, 'classes_'):
            print(f"   Classes do modelo: {list(model.classes_)}")
        if hasattr(model, 'n_features_in_'):
            print(f"   Features esperadas: {model.n_features_in_}")
    
    print(f"   Classes do modelo info: {model_info.get('classes', [])}")
    print("=" * 50)
    
    print("📱 ACESSO LOCAL:")
    print(f"   http://localhost:5000")
    print(f"   http://127.0.0.1:5000")
    print("=" * 50)
    print("🌐 ACESSO NA REDE LOCAL:")
    print(f"   http://{local_ip}:5000")
    print("=" * 50)
    
    # Iniciar câmera
    print("📹 Inicializando câmera...")
    init_camera()
    try:
        # Iniciar Flask com debug habilitado temporariamente
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
        
    except Exception as e:
        print(f"❌ Erro ao iniciar aplicação: {e}")
        traceback.print_exc()