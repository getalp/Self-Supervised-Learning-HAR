
import utils
import csv
import tensorflow as tf

def supervised_downStreamPipeline(fineTuneData,fineTuneLabel,valData,valLabel,testData,testLabel, evaluation_dir,classification_model, FE_Layers, random_FT_weights,trained_FT_weights, finetune_epoch = 50,finetune_batch_size = 64, FT_LR =5e-4):

    macro_f1_list = []
    # Feature Extrator Frozen
    best_validation_weights_dir = evaluation_dir+"Checkpoint_Frozen_FE.h5"
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(best_validation_weights_dir,
        monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True, verbose=0
    )
    classification_model.load_weights(trained_FT_weights)
    
    for model_layer in classification_model.layers[:FE_Layers]:
        model_layer.trainable = False
    
    classification_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FT_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    
    training_history = classification_model.fit(
        x = fineTuneData,
        y = fineTuneLabel,
        batch_size=finetune_batch_size,
        shuffle=True,
        epochs=finetune_epoch,
        callbacks=[best_model_callback],
        verbose=2,
        validation_data=(valData,valLabel)
    )
    classification_model.load_weights(best_validation_weights_dir)
    macro_f1_list.append(utils.getF1Macro(testLabel,classification_model.predict(testData, verbose = 0)))

    utils.plot_learningCurve(training_history,finetune_epoch,evaluation_dir,'Graph_Frozen_FE_')

    # Feature Extrator Unfrozen

    best_validation_weights_dir = evaluation_dir+"Checkpoint_Unfrozen_FE.h5"
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(best_validation_weights_dir,
        monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True, verbose=0
    )
    
    

    for model_layer in classification_model.layers[:FE_Layers]:
        model_layer.trainable = True
    classification_model.load_weights(trained_FT_weights)
    classification_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FT_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    
    training_history = classification_model.fit(
        x = fineTuneData,
        y = fineTuneLabel,
        batch_size=finetune_batch_size,
        shuffle=True,
        epochs=finetune_epoch,
        callbacks=[best_model_callback],
        verbose=2,
        validation_data=(valData,valLabel)
    )
    classification_model.load_weights(best_validation_weights_dir)
    macro_f1_list.append(utils.getF1Macro(testLabel,classification_model.predict(testData, verbose = 0)))


    utils.plot_learningCurve(training_history,finetune_epoch,evaluation_dir,'Graph_Unfrozen_FE_')

    # Feature Extrator Randomly Initialized

    best_validation_weights_dir = evaluation_dir+"Checkpoint_Random_FE.h5"
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(best_validation_weights_dir,
        monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True, verbose=0
    )
    
    classification_model.load_weights(random_FT_weights)

    training_history = classification_model.fit(
        x = fineTuneData,
        y = fineTuneLabel,
        batch_size=finetune_batch_size,
        shuffle=True,
        epochs=finetune_epoch,
        callbacks=[best_model_callback],
        verbose=2,
        validation_data=(valData,valLabel)
    )
    classification_model.load_weights(best_validation_weights_dir)

    macro_f1_list.append(utils.getF1Macro(testLabel,classification_model.predict(testData, verbose = 0)))

    utils.plot_learningCurve(training_history,finetune_epoch,evaluation_dir,'Graph_Random_FE_')
    
    with open(evaluation_dir +'Result_Report.csv','w') as f:
        w = csv.writer(f)
        w.writerow(["Result_Frozen_FE"])
        w.writerow([str(macro_f1_list[0])])
        w.writerow(['Result_Unfrozen_FE'])
        w.writerow([str(macro_f1_list[1])])      
        w.writerow(['Result_Unfrozen_FE_Random'])
        w.writerow([str(macro_f1_list[2])])                    
        
    return macro_f1_list
def downStreamPipeline(fineTuneData,fineTuneLabel,valData,valLabel,testData,testLabel, evaluation_dir,classification_model, FE_Layers, random_FT_weights,trained_FT_weights, finetune_epoch = 50,finetune_batch_size = 64, FT_LR =5e-4):

    macro_f1_list = []
    # Feature Extrator Frozen
    best_validation_weights_dir = evaluation_dir+"Checkpoint_Frozen_FE.h5"
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(best_validation_weights_dir,
        monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True, verbose=0
    )
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15)
    callbacksList = []
    callbacksList.append(stop_early)
    callbacksList.append(best_model_callback)
    stop_early.stopped_epoch

    classification_model.load_weights(trained_FT_weights)
    
    for model_layer in classification_model.layers[:FE_Layers]:
        model_layer.trainable = False
    
    classification_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FT_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    
    training_history = classification_model.fit(
        x = fineTuneData,
        y = fineTuneLabel,
        batch_size=finetune_batch_size,
        shuffle=True,
        epochs=finetune_epoch,
        callbacks= callbacksList,
        verbose=2,
        validation_data=(valData,valLabel)
    )
    classification_model.load_weights(best_validation_weights_dir)
    macro_f1_list.append(utils.getF1Macro(testLabel,classification_model.predict(testData, verbose = 0)))

    plotCurveEpoch = finetune_epoch
    if(stop_early.stopped_epoch != 0):
        plotCurveEpoch = stop_early.stopped_epoch + 1
    
    utils.plot_learningCurve(training_history,plotCurveEpoch,evaluation_dir,'Graph_Frozen_FE_')

    # Feature Extrator Unfrozen

    best_validation_weights_dir = evaluation_dir+"Checkpoint_Unfrozen_FE.h5"
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(best_validation_weights_dir,
        monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True, verbose=0
    )
    
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15)
    callbacksList = []
    callbacksList.append(stop_early)
    callbacksList.append(best_model_callback)


    for model_layer in classification_model.layers[:FE_Layers]:
        model_layer.trainable = True
    classification_model.load_weights(trained_FT_weights)
    classification_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FT_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    
    training_history = classification_model.fit(
        x = fineTuneData,
        y = fineTuneLabel,
        batch_size=finetune_batch_size,
        shuffle=True,
        epochs=finetune_epoch,
        callbacks=callbacksList,
        verbose=2,
        validation_data=(valData,valLabel)
    )
    classification_model.load_weights(best_validation_weights_dir)
    macro_f1_list.append(utils.getF1Macro(testLabel,classification_model.predict(testData, verbose = 0)))


    plotCurveEpoch = finetune_epoch
    if(stop_early.stopped_epoch != 0):
        plotCurveEpoch = stop_early.stopped_epoch + 1
    
    utils.plot_learningCurve(training_history,plotCurveEpoch,evaluation_dir,'Graph_Unfrozen_FE_')

    with open(evaluation_dir +'Result_Report.csv','w') as f:
        w = csv.writer(f)
        w.writerow(["Result_Frozen_FE"])
        w.writerow([str(macro_f1_list[0])])
        w.writerow(['Result_Unfrozen_FE'])
        w.writerow([str(macro_f1_list[1])])      
                 
    return macro_f1_list

