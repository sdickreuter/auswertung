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


def gen_training_data(num, inputs,outputs, filename):
    num_lorentz = int(outputs / 1)
    print(num_lorentz)
    x = np.linspace(0,1,inputs)

    x0s = np.linspace(0, 1, 200)#100)
    sigmas = np.linspace(0.05, 0.8,200)#, 30)
    amps = np.linspace(0.1, 0.8, 200)  # , 30)

    #x0s, sigmas = np.meshgrid(x0s,sigmas)
    #x0s = x0s.ravel()
    #sigmas = sigmas.ravel()

    #num = len(x0s)

    f = open(filename, 'w')
    f.write(str(num) +' '+ str(inputs) +' '+ str(outputs) + "\r\n")

    for i in range(num):
        if num_lorentz > 1:
            n = np.random.randint(1,num_lorentz+1,1)
        else:
            n = 1

        amp = -0.5*np.ones(num_lorentz)
        x0 = -0.5*np.ones(num_lorentz)
        sigma = -0.5*np.ones(num_lorentz)

        for j in range(n):
            #amp[j] = 0.1+np.random.rand(1)*0.8
            amp[j] = amps[np.random.randint(0,len(amps)-1,1)]
            x0[j] = x0s[np.random.randint(0,len(x0s)-1,1)]
            sigma[j] = sigmas[np.random.randint(0, len(sigmas) - 1, 1)]
            #sigma[j] = 0.05+np.random.rand(1)*0.8

        #p = np.concatenate((amp, x0, sigma))

        input = np.zeros(inputs)
        for j in range(n):
            input += lorentz(x,amp[j],x0[j],sigma[j])

        #input = np.gradient(input)
        input = input/np.max(np.abs(input))

        for j in range(inputs):
            f.write(str(input[j]) + " ")

        f.write("\r\n")

        for j in range(num_lorentz):
            #f.write(str(amp[j]) + " ")
            f.write(str(x0[j]) + " ")
            #f.write(str(sigma[j]) + " ")

        f.write("\r\n")
    f.close()
    #plt.plot(input)
    #plt.show()


filename = "training.data"
inputs = 100
outputs = 1 * 2

gen_training_data(5000,inputs, outputs,filename)

nn = libfann.neural_net()
nn.create_shortcut_array((inputs,outputs))

nn.randomize_weights(-0.1,0.1)

nn.cascadetrain_on_file(filename,10000,10,0.0001)

#nn.create_from_file("nn.net")

num_lorentz = int(outputs / 1)

input = [0.5,0.5,0.1,0.3,0.8,0.1]

x = np.linspace(0, 1, inputs)
y = np.zeros(inputs)
for i in range(num_lorentz):
    dy = lorentz(x,input[i*(num_lorentz+1)+0],input[i*(num_lorentz+1)+1],input[i*(num_lorentz+1)+2])
    y += dy


#y = np.gradient(y)
y = y/y.max()

out = nn.run(y)
print(input)
print(out)

y2 = np.zeros(inputs)
for i in range(num_lorentz):
    dy = lorentz(x,input[i*(num_lorentz+1)+0],out[i],input[i*(num_lorentz+1)+2])
    y2 += dy

#y2 = np.gradient(y2)
y = y / np.max(np.abs(y))

plt.plot(x,y)
plt.plot(x,y2,color="black")
plt.title("TRAIN_RPROP")
plt.show()


nn.save("nn.net")



