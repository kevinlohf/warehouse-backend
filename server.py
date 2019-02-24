from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
from starlette.responses import HTMLResponse
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from models.WarehouseItem import WarehouseItem
from werkzeug.utils import secure_filename
from PIL import Image
from object_detector.qr_detector import QRModel
from object_detector.utils import draw_bbs
from pathlib import Path
import os
import io
import ast
import json
import uvicorn
import datetime
import uuid
import pandas as pd


app = Starlette(debug=True, template_directory='templates')
app.add_middleware(CORSMiddleware, allow_origins=['*'])
app.mount('/static', StaticFiles(directory='statics'), name='static')


qr_model = QRModel("object_detector/model.onnx", threshold=0.00)

UPLOAD_FOLDER = 'upload/warehouse/'
OUT_CSV = "upload/output.csv"
DEBUG_FOLDER = 'upload/debug/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

for folder in [UPLOAD_FOLDER, DEBUG_FOLDER]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

WAREHOUSE_DATA = []


@app.route('/')
async def homepage(request):
    template = app.get_template('index.html')
    content = template.render(request=request)
    return HTMLResponse(content)


@app.route('/error')
async def error(request):
    """
    An example error. Switch the `debug` setting to see either tracebacks or 500 pages.
    """
    raise RuntimeError("Oh no")


@app.route("/api/warehouse-item/reset", methods=["GET"])
async def fetch_qr_codes(request):
    response = {'status': 'success'}
    folder = Path(UPLOAD_FOLDER)
    image_paths = folder.glob("*.jpeg")
    for path in image_paths:
        path.unlink()
    folder = Path(DEBUG_FOLDER)
    image_paths = folder.glob("*.jpeg")
    for path in image_paths:
        path.unlink()
    WAREHOUSE_DATA = []
    return JSONResponse(response)


@app.route("/api/warehouse-item/predictions", methods=["GET"])
async def fetch_qr_codes(request):
    response = {'status': 'success'}

    if len(WAREHOUSE_DATA) == 0:
        response["result"] = []
    else:
        df = warehouse_data_to_df(WAREHOUSE_DATA)
        df.time = df.time.apply(lambda x: x.to_datetime64().astype('str'))
        df = df.replace({pd.np.nan: None})
        result = df.to_dict(orient="records")
        response["result"] = result
        # write to file
        df[["location", "productId"]].to_csv(
            OUT_CSV, index=False, header=False)

    return JSONResponse(response)


@app.route("/api/warehouse-item/debug", methods=["GET"])
async def fetch_qr_codes(request):
    response = {'status': 'success'}

    images = [Image.open(path) for path in Path(UPLOAD_FOLDER).glob("*.jpeg")]
    preds = qr_model.predict_batch(images)

    for i in range(len(images)):
        preds = qr_model.predict(images[i])
        bbs = [pred["boundingBox"] for pred in preds]
        image = draw_bbs(images[i], bbs)
        save_upload_image(image, folder=DEBUG_FOLDER)

    return JSONResponse(response)


@app.route("/api/warehouse-item/reload", methods=["GET"])
async def fetch_qr_codes(request):
    response = {'status': 'success'}

    images = [Image.open(path) for path in Path(UPLOAD_FOLDER).glob("*.jpeg")]
    if len(images) == 0:
        return response

    preds = qr_model.predict_batch(images)
    dic = qr_model.create_dict_batch(preds, images)
    WAREHOUSE_DATA.extend(dic)

    return JSONResponse(response)


@app.route("/api/warehouse-item/upload", methods=["POST"])
async def submit_qr_code(request):
    response = {'status': 'success'}
    image_file = await read_upload_file(request)

    if not is_valid_file(image_file):
        response['status'] = 'failure'
        return JSONResponse(response)

    image = await read_upload_image_file(image_file)
    save_upload_image(image)

    predictions = qr_model.predict(image)
    predictions = qr_model.create_dict(predictions, image)
    if len(predictions) > 0:
        WAREHOUSE_DATA.extend(predictions)

    return JSONResponse(response)


@app.exception_handler(404)
async def not_found(request, exc):
    """
    Return an HTTP 404 page.
    """
    template = app.get_template('404.html')
    content = template.render(request=request)
    return HTMLResponse(content, status_code=404)


@app.exception_handler(500)
async def server_error(request, exc):
    """
    Return an HTTP 500 page.
    """
    template = app.get_template('500.html')
    content = template.render(request=request)
    return HTMLResponse(content, status_code=500)


### Util Functions ###


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_valid_file(file):
    if file is None:
        return False
    file_name = secure_filename(file.filename)
    if file_name == '' or not allowed_file(file_name):
        return False
    return True


async def read_upload_file(request):
    form_data = await request.form()
    image_file = form_data['file']
    return image_file


async def read_upload_image_file(file):
    bytes = await (file.read())
    image = Image.open(io.BytesIO(bytes))
    return image


def save_upload_image(image, folder=UPLOAD_FOLDER):
    current_time = datetime.datetime.now().strftime("%H-%M-%S")
    save_file_name = f"{current_time}-{uuid.uuid1()}.jpeg"
    save_file_path = folder + save_file_name
    image.save(save_file_path, "jpeg")


def warehouse_data_to_df(warehouse):

    warehouse = pd.DataFrame.from_dict(warehouse)
    counts = warehouse.groupby("location").productId.value_counts(
    ).to_frame().rename(columns={"productId": "counts"}).reset_index()
    most_frequent_idx = counts.groupby(
        "location", as_index=False).counts.idxmax().values
    latest_time = warehouse.groupby("location", as_index=False).time.max()
    most_frequent = counts.iloc[most_frequent_idx]
    results = pd.merge(most_frequent, latest_time, how="outer")
    return results.drop(columns=["counts"])


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=3000, timeout_keep_alive=60)
