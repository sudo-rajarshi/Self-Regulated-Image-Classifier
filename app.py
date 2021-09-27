from flask import Flask, render_template, request, redirect
from sric_lib import *
import shutil
from datetime import datetime
from google_drive_downloader import GoogleDriveDownloader as gdd
from flask_ngrok import run_with_ngrok
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders 
import webbrowser
from threading import Timer


app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1
app.config["SECRET_KEY"] = 'dsds'

@app.route('/', methods=['GET','POST'])
def main():
    return render_template('source.html')



@app.route('/source', methods=['GET','POST'])
def source():
    if request.method == 'POST':
        global now, glink, gid, source_address, toaddr
        global source_data, class_no, classes, source_folder_dir_class, smallest_h, smallest_w
        global directory, train_directory, validation_directory, test_directory
        global name_res_dir

        now = datetime.now()

        # SOURCE
        glink = request.form.get('glink')
        toaddr = request.form.get('toaddr')

        gid = glink[32:65]
        print(gid)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        source_address = os.path.join(dir_path, 'assets')
        if os.path.exists(source_address):
            pass
        else:
            os.mkdir(source_address)

        gdd.download_file_from_google_drive(file_id = gid,
                                            dest_path=os.path.join(source_address, 'train.zip'),
                                            unzip=True,
                                            showsize=True,
                                            overwrite=True) 

        
        source_data, class_no, classes, source_folder_dir_class, smallest_h, smallest_w = analyze(source_address)
        directory, train_directory, validation_directory, test_directory = mk_splitted_dir(source_address)

        name_res_dir = toaddr.split('@')[0]

        data = np.array([source_data, class_no, name_res_dir])

    return render_template('split.html', data = data)



@app.route('/split', methods=['GET','POST'])
def split():
    if request.method == 'POST':
        global val_size, test_size, train_size, train_samples, test_samples, val_samples, height, width

        if request.form.get('val_size') == '':
            val_size = 0.15
        else:
            val_size = float(request.form.get('val_size'))
            val_size = val_size/100

        print(val_size)

        if request.form.get('test_size') == '':
            test_size = 0.10
        else:
            test_size = float(request.form.get('test_size'))
            test_size = test_size/100

        splitted(class_no, classes, train_directory, validation_directory, test_directory, source_folder_dir_class, val_size, test_size)
        train_size = 1-(test_size + val_size)
        train_samples = source_data*train_size
        test_samples = source_data*test_size
        val_samples = source_data*val_size
        
        height, width = get_size(directory)

        data = np.array([
                        int(smallest_h),
                        int(smallest_w),
                        int(np.mean(height)),
                        int(np.mean(width)),
                        name_res_dir
                        ])

    return render_template('preprocess.html', data = data) 



@app.route('/preprocess', methods=['GET','POST'])
def preprocess():
    if request.method == 'POST':
        global img_w, img_h, dim
        global char, char_subfolder, char_name

        if request.form.get('img_w') == '':
            img_w = 128
        else:
            img_w = int(request.form.get('img_w'))

        if request.form.get('img_h') == '':
            img_h = 128
        else:
            img_h = int(request.form.get('img_h'))

        dim = resize(directory, img_w, img_h)
        char, char_subfolder, char_name = monitor_dir(now, name_res_dir, source_address)

    return render_template('hyper_parameter.html', name_res_dir = name_res_dir)



@app.route('/hyper_parameter', methods=['GET','POST'])
def hyper_parameter():
    if request.method == 'POST':
        global learning_rate, epoch, batch_size

        if request.form.get('learning_rate') == '':
            learning_rate = 0.001
        else:
            learning_rate = int(request.form.get('learning_rate'))

        if request.form.get('epoch') == '':
            epoch = 25
        else:
            epoch = int(request.form.get('epoch'))

        if request.form.get('batch_size') == '':
            batch_size = 8
        else:
            batch_size = int(request.form.get('batch_size'))

    return render_template('early_stop.html', name_res_dir = name_res_dir)



@app.route('/early_stop', methods=['GET','POST'])
def early_stop():
    if request.method == 'POST':
        global char, char_subfolder, char_name
        global early_stop, monitor, patience, callback, best_model_address
        global output_activation, losses, class_mode, output_layer
        global train_generator, validation_generator, test_generator, train_base_model

        char, char_subfolder, char_name = monitor_dir(now, name_res_dir, source_address)

        # EARLY STOP
        early_stop = request.form.get('early_stop')
        monitor = request.form.get('monitor')

        if request.form.get('patience') == '':
            patience = 5
        else:
            patience = int(request.form.get('patience'))

        callback, best_model_address = monitor_metric('y', early_stop, monitor, patience, char)
        output_activation, losses, class_mode, output_layer = essentials(class_no)
        train_generator, validation_generator, test_generator = generators(batch_size, class_mode, dim, train_directory, validation_directory, test_directory)
        
        if request.form.get('train_base_model') == '':
            train_base_model = 'n'
        else:
            train_base_model = request.form.get('train_base_model')

        data = np.array([
                        int(img_w), 
                        int(img_h), 
                        learning_rate, 
                        int(batch_size),
                        name_res_dir
                        ])

    return render_template('model.html', data = data)



@app.route('/model', methods=['GET','POST'])
def model():
    if request.method == 'POST':
        global tl_models, dense, dropout, model
        global layer, layer, conv_layer, conv_layer, conv, conv_size
        global optimizer_select, optimizer, comped_model
        global history, duration, train_score, val_score, test_score
        global acc, req_epochs
        global training_accuracy, validation_accuracy, test_accuracy, test_precision, test_recall
        global y_true, y_pred, labels, cm, classification_reports, sensitivity, specificity
        global char_name_zip, char_zip, toaddr

        tl_models = request.form.get('tl_models')

        if request.form.get('dense') == '':
            dense = 256
        else:
            dense = int(request.form.get('dense'))

        if request.form.get('dropout') == '':
            dropout = 0.30
        else:
            dropout = int(request.form.get('dropout'))
            dropout = dropout/100

        if request.form.get('train_base_model') == '':
            train_base_model = 'N'
        else:
            train_base_model = str(request.form.get('train_base_model'))
        
        if tl_models == 'Basic': 
            model = Custom_Prebuilt_Model(dim, output_layer, output_activation)

        if tl_models == 'VGG16': 
            model= vgg16(train_base_model, dim, dense, dropout, output_layer, output_activation)

        elif tl_models == 'VGG19':
            model = vgg19(train_base_model, dim, dense, dropout, output_layer, output_activation)

        elif tl_models == 'MobileNet':
            model = MobileNet(train_base_model, dim, dense, dropout, output_layer, output_activation)

        elif tl_models == 'Inception':
            model = InceptionV3(train_base_model, dim, dense, dropout, output_layer, output_activation) 

        elif tl_models == 'ResNet50':
            model = ResNet50(train_base_model, dim, dense, dropout, output_layer, output_activation)

        elif tl_models == 'Own':
            if request.form.get('layer') == '':
                layer = 5
            else:
                layer = int(request.form.get('layer'))

            if request.form.get('conv_layer') == '':
                conv_layer = 1
            else:
                conv_layer = int(request.form.get('conv_layer'))

            if request.form.get('conv') == '':
                conv = 16
            else:
                conv = int(request.form.get('conv'))

            if request.form.get('conv_size') == '':
                conv_size = 3
            else:
                conv_size = int(request.form.get('conv_size'))

            model = Custom_Model(dim, layer, conv_layer, conv, conv_size, dense, dropout, output_layer, output_activation)

        optimizer_select = request.form.get('optimizer')
        optimizer = optimizer_selection('y', optimizer_select, learning_rate)

        comped_model = model_compile(model, optimizer, losses)

        history, duration, train_score, val_score, test_score = train(physical_devices, comped_model, train_generator, validation_generator, test_generator, epoch, callback)
        acc=history.history['accuracy']
        req_epochs = len(acc)

        print("Execution Time: {} seconds".format(duration))
        print('Model created at {}'.format(source_address))

        characteristics(history, char)

        training_accuracy, validation_accuracy, test_accuracy, test_precision, test_recall = performance(train_score, val_score, test_score)
        print("The training accuracy is: " + str(training_accuracy) + ' %')
        print("The validation accuracy is: " + str(validation_accuracy) + ' %')
        print("The test accuracy is: " + str(test_accuracy) + ' %')
        print("Test Precision: {}".format(test_precision))
        print("Test Recall: {}".format(test_recall))
        

        y_true, y_pred, labels = pred(test_directory, test_generator, class_no, best_model_address, dim)
        cm, classification_reports, sensitivity, specificity = report(y_true, y_pred, labels)
        conf_mat(cm, labels, char)
        
        save_readme(char,
                class_no,
                train_size,
                val_size,
                test_size,
                img_w,
                img_h,
                learning_rate,
                early_stop, 
                epoch,
                patience,
                req_epochs,
                batch_size,
                monitor,
                dropout,
                output_activation,
                losses,
                optimizer_select,
                tl_models,
                model,
                test_accuracy,
                test_precision,
                test_recall,
                classification_reports,
                sensitivity,
                specificity,
                duration
                )

        shutil.make_archive(char, 'zip', char)
        char_name_zip = char_name + '.zip'
        char_zip = char + '.zip'

        fromaddr = "email@gmail.com"

        data = np.array([
                        glink, 
                        source_data, 
                        train_size*100,
                        int(train_samples),
                        val_size*100, 
                        int(val_samples),
                        test_size*100, 
                        int(test_samples),
                        img_w, 
                        img_h, 
                        learning_rate, 
                        epoch, 
                        batch_size, 
                        early_stop, 
                        monitor, 
                        patience, 
                        req_epochs,
                        round(training_accuracy, 2),
                        round(validation_accuracy, 2),
                        round(test_accuracy, 2),
                        round(test_precision, 2),
                        round(test_recall,2),
                        round(sensitivity*100,2),
                        round(specificity*100,2),
                        int(duration),
                        toaddr,
                        fromaddr,
                        name_res_dir
                        ])

        msg = MIMEMultipart() 

        msg['From'] = fromaddr
        msg['To'] = toaddr 

        msg['Subject'] = "Congratulations! Your model is trained."
        body = "Please download the attachment below."

        msg.attach(MIMEText(body, 'plain')) 

        filename = char_name_zip
        attachment = open(char_zip, "rb") 

        p = MIMEBase('application', 'octet-stream') 
        p.set_payload((attachment).read()) 
        encoders.encode_base64(p) 
        p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 
        msg.attach(p) 
        
        s = smtplib.SMTP('smtp.gmail.com', 587) 
        s.starttls() 
        s.login(fromaddr, "password") 
        text = msg.as_string() 
        s.sendmail(fromaddr, toaddr, text) 
        s.quit() 

        shutil.rmtree(char)

    return render_template('summary.html', data=data)


def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')


if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run()
