from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
import scipy.ndimage as ndimage  # مكتبة التوسيط الدقيق

app = Flask(__name__)

try:
    model = load_model('model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(img):
    """دالة احترافية لقص الرقم وتوسيطه باستخدام مركز الكتلة (Center of Mass)"""
    # 1. قص الأطراف السوداء للوصول إلى الرقم فقط
    bbox = img.getbbox()
    if not bbox:
        return None # اللوحة فارغة
    img = img.crop(bbox)

    # 2. تغيير الحجم مع الحفاظ على النسبة (أطول ضلع = 20 بكسل)
    width, height = img.size
    if width == 0 or height == 0: return None
    aspect_ratio = width / height
    
    if width > height:
        new_width = 20
        new_height = int(20 / aspect_ratio)
    else:
        new_height = 20
        new_width = int(20 * aspect_ratio)
        
    img = img.resize((new_width, new_height), Image.LANCZOS)

    # 3. وضع الرقم في صورة 28x28 مؤقتة
    canvas = Image.new('L', (28, 28), 0)
    paste_x = (28 - new_width) // 2
    paste_y = (28 - new_height) // 2
    canvas.paste(img, (paste_x, paste_y))

    # 4. التوسيط الدقيق باستخدام مركز الكتلة (هذا هو السر!)
    arr = np.array(canvas).astype('float32')
    cy, cx = ndimage.center_of_mass(arr)
    rows, cols = arr.shape
    shiftx = np.ceil(cols / 2.0 - cx).astype(int)
    shifty = np.ceil(rows / 2.0 - cy).astype(int)
    arr = ndimage.shift(arr, shift=(shifty, shiftx))

    return Image.fromarray(arr)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json
        image_data = data['image'].split(",")[1]
        image_bytes = base64.b64decode(image_data)

        # لوحة الرسم ترسم أبيض على أسود، وبيانات MNIST أبيض على أسود. لا نقلب الألوان!
        img = Image.open(io.BytesIO(image_bytes)).convert('L')

        # تطبيق المعالجة
        processed_img = preprocess_image(img)
        if processed_img is None:
            return jsonify({'error': 'اللوحة فارغة، يرجى رسم رقم أولاً'}), 400

        # تحويل الصورة إلى مصفوفة وتطبيعها
        img_array = np.array(processed_img).astype('float32') / 255
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        return jsonify({'digit': predicted_digit, 'confidence': round(confidence, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)