# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:23:43 2023

@author: hizza
"""
import keras
import sys #sys only needed if running in command line
import h5py
import numpy as np
#from keras import backend as K
#import tensorflow as tf

"""
#Uncomment when using command line
clean_data_filename = str(sys.argv[1])
poisoned_data_filename = str(sys.argv[2])
model_filename = str(sys.argv[3])
prune = str(sys.argv[4])
goodnet_percentage = str(sys.argv[5])
"""

#Uncomment for use within IDE
clean_data_filename = "./data/cl/valid.h5"
clean_data_final = "./data/cl/test.h5"
poisoned_data_filename = "./data/bd/bd_valid.h5"
poisoned_data_final = "./data/bd/bd_test.h5"
model_filename = "./models/bd_net.h5"
prune = "target" #target for only pruning channels that impact backdoor performance, final for just the final layer
goodnet_percentage = 10  #amount of accuracy loss allowed

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def main():
    model_path = "./models/gn-x{perc}.h5" #where to save the model
    cl_x_test, cl_y_test = data_loader(clean_data_filename)
    bd_x_test, bd_y_test = data_loader(poisoned_data_filename)
    cl_x_final, cl_y_final = data_loader(clean_data_final)
    bd_x_final, bd_y_final = data_loader(poisoned_data_final)
    bd_model = keras.models.load_model(model_filename)
    
    
    
    bd_model.summary()
    num_classes = np.unique(cl_y_test).shape[0]
    print("Number of Classes: ",num_classes)
    print(np.unique(cl_y_test))
    cl_label_p = np.argmax(bd_model.predict(cl_x_test), axis=1)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test))*100
    print('Original Clean Classification Validation accuracy:', clean_accuracy)
    
    bd_label_p = np.argmax(bd_model.predict(bd_x_test), axis=1)
    asr = np.mean(np.equal(bd_label_p, bd_y_test))*100
    print('Original Attack Validation Success Rate:', asr)
    
    
    cl_label_final = np.argmax(bd_model.predict(cl_x_final), axis=1)
    clean_accuracy_final = np.mean(np.equal(cl_label_final, cl_y_final))*100
    print('Original Clean Classification Test accuracy:', clean_accuracy_final)
    
    bd_label_final = np.argmax(bd_model.predict(bd_x_final), axis=1)
    asr = np.mean(np.equal(bd_label_final, bd_y_final))*100
    print('Original Attack Test Success Rate:', asr)
    
    
    
    #method to prune just the final layer
    if (prune == "final"):
        gn_x2 = keras.models.load_model(model_filename)
        acc_curr = clean_accuracy
        it_2 = 79
        while acc_curr >= (clean_accuracy-goodnet_percentage):
            layer_curr = gn_x2.layers[7].get_weights()
            layer_curr[0][:,:,:,it_2]=0
            gn_x2.layers[7].set_weights(layer_curr)
            #print(gn_x2.layers[7].get_weights())
            labels_curr = np.argmax(gn_x2.predict(cl_x_test), axis=1)
            acc_curr = np.mean(np.equal(labels_curr, cl_y_test))*100
            if it_2%10 == 0:
                print(acc_curr)
            del(layer_curr)
            del(labels_curr)
            it_2 = it_2-1
        cl_label2 = np.argmax(gn_x2.predict(cl_x_test), axis=1)
        clean_accuracy2 = np.mean(np.equal(cl_label2, cl_y_test))*100
        print(f"Goodnet {goodnet_percentage}% Clean Classification Validation accuracy: {clean_accuracy2}")
        pred = np.argmax(gn_x2.predict(bd_x_test), axis=1)
        for i in range(len(pred)):
            if pred[i] != bd_label_p[i]:
                pred[i] = num_classes
        asr2 = np.mean(np.equal(pred, bd_y_test))*100
        print(f"Goodnet {goodnet_percentage}% Attack Validation Success Rate: {asr2}")
        
        cl_label2 = np.argmax(gn_x2.predict(cl_x_final), axis=1)
        clean_accuracy2 = np.mean(np.equal(cl_label2, cl_y_final))*100
        print(f"Goodnet {goodnet_percentage}% Clean Classification Test accuracy: {clean_accuracy2}")
        pred = np.argmax(gn_x2.predict(bd_x_final), axis=1)
        for i in range(len(pred)):
            if pred[i] != bd_label_final[i]:
                pred[i] = num_classes
        asr2 = np.mean(np.equal(pred, bd_y_final))*100
        print(f"Goodnet {goodnet_percentage}% Attack Test Success Rate: {asr2}")
        del(gn_x2)

    #method to do more effictive targeted pruning
    if prune == "target":
        gn_x10 =  keras.models.load_model(model_filename)
        acc_curr = clean_accuracy
        it_10 = 0
        layer_ind = 7 #if you would like to change the starting layer, you can change this to 5
        pad = 0
        if goodnet_percentage == 4:
           pad = 0.14
        if goodnet_percentage ==10:
            pad = 0.3
        while acc_curr >= (clean_accuracy-goodnet_percentage+pad):
            layer_curr = gn_x10.layers[layer_ind].get_weights()
            gn_temp = keras.models.load_model(model_filename)
            #ind = np.argmax()
            layer_curr[0][:,:,:,it_10]=0
            gn_temp.layers[layer_ind].set_weights(layer_curr)
            temp_labels = np.argmax(gn_temp.predict(cl_x_test), axis=1)
            temp_acc = np.mean(np.equal(temp_labels, cl_y_test))*100
            pred = np.argmax(gn_temp.predict(bd_x_test), axis=1)
            for i in range(len(pred)):
                if pred[i] != bd_label_p[i]:
                    pred[i] = num_classes
            asr10 = np.mean(np.equal(pred, bd_y_test))*100
            if asr10<100 and temp_acc >= (clean_accuracy-goodnet_percentage):
                gn_x10.layers[layer_ind].set_weights(layer_curr)
            labels_curr = np.argmax(gn_x10.predict(cl_x_test), axis=1)
            acc_curr = np.mean(np.equal(labels_curr, cl_y_test))*100
            #print accuracy every 10 steps, see how the final model is doing
            if it_10%10 == 0:
                print("Step accuracy: ",acc_curr)
            elif it_10 ==79 and layer_ind == 7:
                layer_ind = 5
                it_10 = 0
            elif it_10 == 59 and layer_ind == 5:
                layer_ind = 3
                it_10 = 0
            elif it_10 == 39 and layer_ind ==3:
                break
            del(gn_temp)
            del(temp_labels)
            del(layer_curr)
            del(labels_curr)
            it_10 = it_10+1
            
        cl_label10 = np.argmax(gn_x10.predict(cl_x_test), axis=1)
        clean_accuracy10 = np.mean(np.equal(cl_label10, cl_y_test))*100
        print(f"Goodnet {goodnet_percentage}% Clean Classification Validation accuracy: {clean_accuracy10}")
        pred = np.argmax(gn_x10.predict(bd_x_test), axis=1)
        for i in range(len(pred)):
            if pred[i] != bd_label_p[i]:
                pred[i] = num_classes
        asr10 = np.mean(np.equal(pred, bd_y_test))*100
        print(f"Goodnet {goodnet_percentage}% Attack Validation Success Rate: {asr10}")
        
        cl_label10 = np.argmax(gn_x10.predict(cl_x_final), axis=1)
        clean_accuracy10 = np.mean(np.equal(cl_label10, cl_y_final))*100
        print(f"Goodnet {goodnet_percentage}% Clean Classification Test accuracy: {clean_accuracy10}")
        pred = np.argmax(gn_x10.predict(bd_x_final), axis=1)
        for i in range(len(pred)):
            if pred[i] != bd_label_final[i]:
                pred[i] = num_classes
        asr10 = np.mean(np.equal(pred, bd_y_final))*100
        print(f"Goodnet {goodnet_percentage}% Attack Test Success Rate: {asr10}")
        gn_x10.save(model_path.format(perc = goodnet_percentage))
        del(gn_x10)
    
if __name__ == '__main__':
        main()