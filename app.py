from flask import Flask, request, jsonify, render_template
import os
from utils.parse import *

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    sample_data = {"message": "Hello, this is a sample API response"}
    return jsonify(sample_data)

@app.route('/api/data', methods=['POST'])
def post_data():
    data = request.get_json()
    return jsonify({"received_data": data}), 201

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    # log 1: before optimization
    modify_ltspice_lib_path(file_path)
    run_ltspice(file_path, './logs/before/trans_logs')
    # log 2: before optimization
    convert_tran_ac(file_path)
    run_ltspice(file_path, './logs/before/ac_logs')
    
    key = "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"
    
    optimized_params = optimize_circuit_parameters("./uploads/Common_Source_amp.txt", key)
    print(optimized_params)


    update_netlist("./uploads/Common_Source_amp.txt", "./uploads/Modified_Common_Source_amp.txt", optimized_params)
    modified_file_path = "./uploads/Modified_Common_Source_amp.txt"

    reverse_convert_tran_ac(modified_file_path)
    run_ltspice(modified_file_path, './logs/after/trans_logs')
    convert_tran_ac(modified_file_path)
    run_ltspice(modified_file_path, './logs/after/ac_logs')

    gain_db_before = read_value_from_log("./logs/before/ac_logs/Common_Source_amp.log", "gain_db:")
    gain_db_after = read_value_from_log("./logs/after/ac_logs/Modified_Common_Source_amp.log", "gain_db:")
    
    p_r5_before = read_value_from_log("./logs/before/trans_logs/Common_Source_amp.log", "p_r5:")
    p_r5_after = read_value_from_log("./logs/after/trans_logs/Modified_Common_Source_amp.log", "p_r5:")
    
    return render_template('index.html', 
                           gain_db_before=gain_db_before, gain_db_after=gain_db_after,
                           p_r5_before=p_r5_before, p_r5_after=p_r5_after)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
