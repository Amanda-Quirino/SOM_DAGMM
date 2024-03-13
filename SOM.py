import numpy as np
from minisom import MiniSom

#Função responsável por criar o SOM e treina-lo
def som_train(data, x=10, y=10, sigma=1, learning_rate= 0.8, iters= 10000, neighborhood_function= 'gaussian'):
    input_len = data.shape[1]
    print("SOM training started:")
    som = MiniSom(x= x, y= y, input_len=input_len, sigma=sigma, learning_rate=learning_rate, neighborhood_function=neighborhood_function)
    som.random_weights_init(data)
    som.train_random(data, iters)
    return som

#Função responsável por realizar a predição do modelo som
def som_pred(som_model, data, outlier_percentage):
    model = som_model
    data = data.numpy()
    quantization_errors = np.linalg.norm(model.quantization(data) - data, axis=1)
    error_threshold = np.percentile(quantization_errors, 100*(1-outlier_percentage)+5)
    is_anomaly = quantization_errors > error_threshold
    y_pred = np.multiply(is_anomaly, 1)
    return y_pred