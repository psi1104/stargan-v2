import os
import uuid

import torch
from flask import Flask, render_template, request, jsonify, send_from_directory
from munch import Munch
from torch.backends import cudnn
from werkzeug.utils import secure_filename

from core.data_loader import get_test_loader
from core.solver import Solver
from core.wing import align_faces
from main import parse_args

args = parse_args()

UPLOAD_FOLDER = 'upload'

cudnn.benchmark = True
torch.manual_seed(args.seed)

app = Flask(__name__, template_folder="./templates/")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

@app.route('/CelebA_HQ', methods=['POST'])
def CelebA_HQ():
    input_file = request.files['source']
    if not input_file:
        return jsonify({'message': 'nofile'}), 400
    if input_file.content_type not in ['image/jpeg', 'image/jpg', 'image/png']:
        return jsonify({'message': 'only support jpeg, jpg or png'}), 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    f_id = str(uuid.uuid4())

    fname = secure_filename(f_id + input_file.filename)

    input_file.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))

    args.out_dir = 'assets/representative/celeba_hq/src/' + f_id + '/' + f_id

    align_faces(args, UPLOAD_FOLDER, args.out_dir)

    args.src_dir = 'assets/representative/celeba_hq/src/' + f_id

    loaders = Munch(src=get_test_loader(root=args.src_dir,
                                    img_size=args.img_size,
                                    batch_size=args.val_batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers),
                ref=get_test_loader(root=args.ref_dir,
                                    img_size=args.img_size,
                                    batch_size=args.val_batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers))
    ###
    args.checkpoint_dir = 'expr/checkpoints/celeba_hq'

    result_dir = 'expr/results/' + f_id
    args.result_dir = result_dir

    solver = Solver(args)
    solver.sample(loaders)

    result = send_from_directory(directory=result_dir, filename='reference.jpg')


    return result


@app.route('/health')
def health():
    return "ok"

@app.route('/')
def main(): 
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, port=8104, host='0.0.0.0', threaded=True)