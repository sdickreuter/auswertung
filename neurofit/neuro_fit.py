# int main()
# {
#     const unsigned int num_input = 2;
#     const unsigned int num_output = 1;
#     const unsigned int num_layers = 3;
#     const unsigned int num_neurons_hidden = 3;
#     const float desired_error = (const float) 0.001;
#     const unsigned int max_epochs = 500000;
#     const unsigned int epochs_between_reports = 1000;
#
#     struct fann *ann = fann_create_standard(num_layers, num_input,
#         num_neurons_hidden, num_output);
#
#     fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
#     fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
#
#     fann_train_on_file(ann, "xor.data", max_epochs,
#         epochs_between_reports, desired_error);
#
#     fann_save(ann, "xor_float.net");
#
#     fann_destroy(ann);
#
#     return 0;
#
# #include <stdio.h>
# #include "floatfann.h"
#
# int main()
# {
#     fann_type *calc_out;
#     fann_type input[2];
#
#     struct fann *ann = fann_create_from_file("xor_float.net");
#
#     input[0] = -1;
#     input[1] = 1;
#     calc_out = fann_run(ann, input);
#
#     printf("xor test (%f,%f) -> %f\n", input[0], input[1], calc_out[0]);
#
#     fann_destroy(ann);
#     return 0;
# }


from fann2 import libfann
import matplotlib.pyplot as plt
import numpy as np

def lorentz(x, amplitude, xo, sigma):
    xo = float(xo)
    g = amplitude * np.power(sigma / 2, 2.) / (np.power(sigma / 2, 2.) + np.power(x - xo, 2.))
    return g.ravel()


def gen_training_data_rand(num, inputs,outputs, filename):
    num_lorentz = int(outputs / 2)

    x = np.linspace(0,1,inputs)

    f = open(filename, 'w')
    f.write(str(num) +' '+ str(inputs) +' '+ str(outputs) + "\r\n")

    for i in range(num):
        if num_lorentz > 1:
            n = np.random.randint(1,num_lorentz,1)
        else:
            n = 1
        amp = np.zeros(num_lorentz)
        x0 = np.zeros(num_lorentz)
        sigma = np.zeros(num_lorentz)

        amp[range(n)] = 0.1+np.random.rand(n)*0.8
        x0[range(n)] = 0+np.random.rand(n)*1
        sigma[range(n)] = 0.01+np.random.rand(n)*0.8
        p = np.concatenate((amp, x0, sigma))

        input = np.zeros(inputs)
        for j in range(n):
            input += lorentz(x,amp[j],x0[j],sigma[j])

        input = np.gradient(input)

        for j in range(inputs):
            f.write(str(input[j]) + " ")

        f.write("\r\n")

        for j in range(num_lorentz):
            #f.write(str(amp[j]) + " ")
            f.write(str(x0[j]) + " ")
            f.write(str(sigma[j]) + " ")

        f.write("\r\n")
    f.close()
    #plt.plot(input)
    #plt.show()

def gen_training_data(inputs,outputs, filename):
    num_lorentz = int(outputs / 2)

    x = np.linspace(0,1,inputs)

    x0s = np.linspace(0, 1, 500)#100)
    sigmas = np.linspace(0.05, 0.8,100)#, 30)

    x0s, sigmas = np.meshgrid(x0s,sigmas)
    x0s = x0s.ravel()
    sigmas = sigmas.ravel()

    num = len(x0s)

    f = open(filename, 'w')
    f.write(str(num) +' '+ str(inputs) +' '+ str(outputs) + "\r\n")

    for i in range(num):
        if num_lorentz > 1:
            n = np.random.randint(1,num_lorentz,1)
        else:
            n = 1
        amp = np.zeros(num_lorentz)
        x0 = np.zeros(num_lorentz)
        sigma = np.zeros(num_lorentz)

        amp[:n] = 0.1+np.random.rand(1)*0.8
        x0[:n] = x0s[i]
        sigma[:n] = sigmas[i]

        p = np.concatenate((amp, x0, sigma))

        input = np.zeros(inputs)
        for j in range(n):
            input += lorentz(x,amp[j],x0[j],sigma[j])

        input = np.gradient(input)

        for j in range(inputs):
            f.write(str(input[j]) + " ")

        f.write("\r\n")

        for j in range(num_lorentz):
            #f.write(str(amp[j]) + " ")
            f.write(str(x0[j]) + " ")
            f.write(str(sigma[j]) + " ")

        f.write("\r\n")
    f.close()
    #plt.plot(input)
    #plt.show()


filename = "training_.data"
inputs = 100
outputs = 1 * 2

#gen_training_data_rand(2000, inputs, outputs,filename)
gen_training_data(inputs, outputs,filename)


learning_rate = 0.7

nn = libfann.neural_net()
nn.create_standard_array((inputs,int(inputs*0.7),int(inputs*0.3),outputs))

#nn.set_training_algorithm(libfann.TRAIN_QUICKPROP)
nn.set_training_algorithm(libfann.TRAIN_RPROP)
nn.set_learning_rate(learning_rate)

nn.set_activation_steepness_hidden(1.0)
nn.set_activation_steepness_output(1.0)

nn.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC_STEPWISE)
nn.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

nn.randomize_weights(0.01,0.1)

print("neurons: "+ str(nn.get_total_neurons()))

nn.train_on_file(filename,20000,500,0.0001)

#nn.create_from_file("nn_.net")

input = [0.5,0.5,0.1]
#input = [0.232679774849, 0.711396909983, 0.658477352212]
print(input)
x = np.linspace(0, 1, inputs)
y = lorentz(x,input[0],input[1],input[2])
y = np.gradient(y)
out = nn.run(y)
print(out)
plt.plot(x,y)
plt.plot(x,np.gradient(lorentz(x,0.5,out[0],out[1])))
plt.title("TRAIN_RPROP")
plt.show()


nn.save("nn_.net")



