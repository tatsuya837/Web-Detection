from flask import Flask,render_template,request

from detection import load_model,load_image_into_numpy_array,run_inference_for_single_image,show_inference,category_index

import base64
import os

from io import BytesIO
# from gtts import gTTS
# from playsound import playsound

app = Flask(__name__)
app.config.from_object(__name__)

# アップロード先のフォルダを指定
# UPLOAD_FOLDER = 'detection/static/image/'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    """ 検索画面 """
    return render_template('index.html', results={})

@app.route('/load', methods=['POST'])
def ajax_load():
    # model_name = 'ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8'

    model_name = request.form['model_name']
    global detection_model
    detection_model = load_model(model_name)
    return model_name

# @app.route('/upload', methods=['POST'])
# def upload():

#     img_file = request.files['img_select']

#     # 画像のアップロード先URLを生成する
#     img_url = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)

#     # 画像をアップロード先に保存する
#     img_file.save(img_url)

#     print('img_url',img_url)

#     # 画像をWEBページに表示する
#     return render_template('index.html', img_url=img_url, img_src=img_url)

@app.route('/detection', methods=['POST'])
def detection():
    """ 物体検出画面 """
    image_np = load_image_into_numpy_array(request.form['img_url'])

    # Actual detection.
    output_dict = run_inference_for_single_image(detection_model, image_np)

    # Visualization of the results of a detection.
    dect_img = show_inference(image_np, output_dict)

    # 物体検出データの抽出
    classes = output_dict['detection_classes']
    scores = output_dict['detection_scores']
    boxes = output_dict['detection_boxes']
    width, height = dect_img.size

    global N_classes,N_scores,N_boxes
    N_classes,N_scores,N_boxes = [],[],[]

    for cls_id,scr_id,box_id in zip(classes, scores, boxes):
        if scr_id >= 0.35:

            N_classes.append(category_index[cls_id]['name'])
            N_scores.append(scr_id)
            N_boxes.append(box_id * [height, width, height, width])

    print('N_classes',N_classes)
    print('N_scores',N_scores)
    print('N_boxes',N_boxes)

    buffer = BytesIO() # メモリ上への仮保管先を生成
    dect_img.save(buffer, format="PNG") # pillowのImage.saveメソッドで仮保管先へ保存
    base64img = base64.b64encode(buffer.getvalue()).decode().replace("'", "")
    
    base64data = "data:image/png;base64,{}".format(base64img)
    
    return render_template('detection.html', dect_img=base64data)

@app.route('/detection/img', methods=['POST'])
def ajax_img():

    woffsetX = int(request.form['offsetX'])
    woffsetY = int(request.form['offsetY'])

    print(woffsetX,woffsetY)
    wobj = ''

    for cls_id,box_id in zip(N_classes, N_boxes):
        y_min, x_min, y_max, x_max = box_id
        if (y_min < woffsetY < y_max):
            if (x_min < woffsetX < x_max):
                wobj += cls_id

    if wobj != '':
        print(wobj)
        # wtts = gTTS(text=wobj, lang="en", slow=True)
        # wtts.save("object.mp3")
        # playsound("object.mp3")
        # os.remove("object.mp3")

    return wobj

if __name__ == '__main__':
    app.run()