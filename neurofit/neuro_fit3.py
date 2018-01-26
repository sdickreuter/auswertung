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

        input = np.gradient(input)
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


learning_rate = 0.7

nn = libfann.neural_net()
#nn.create_standard_array((inputs,int(inputs*3.0),int(inputs*1.0),int(inputs*0.2),outputs))
nn.create_shortcut_array((inputs,int(inputs*3.0),int(inputs*1.0),int(inputs*0.2),outputs))
#nn.create_shortcut_array((inputs,int(inputs*1.0),int(inputs*0.5),outputs))
#nn.create_standard_array((inputs,int(inputs*2.0),outputs))

#nn.set_training_algorithm(libfann.TRAIN_QUICKPROP)
nn.set_training_algorithm(libfann.TRAIN_RPROP)
nn.set_learning_rate(learning_rate)

nn.set_activation_steepness_hidden(1.0)
nn.set_activation_steepness_output(1.0)
#nn.set_activation_steepness_hidden(0.2)
#nn.set_activation_steepness_output(0.2)
nn.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC_STEPWISE)
nn.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
#nn.set_activation_function_hidden(libfann.LINEAR_PIECE_SYMMETRIC)
#nn.set_activation_function_output(libfann.LINEAR_PIECE_SYMMETRIC)

nn.randomize_weights(-0.5,0.5)

print("neurons: "+ str(nn.get_total_neurons()))

nn.train_on_file(filename,50000,250,0.0001)

#nn.create_from_file("nn.net")

num_lorentz = int(outputs / 1)

input = [0.5,0.5,0.1,0.3,0.8,0.1]


x = np.linspace(0, 1, inputs)
y = np.zeros(inputs)
for i in range(num_lorentz):
    dy = lorentz(x,input[i*(num_lorentz+1)+0],input[i*(num_lorentz+1)+1],input[i*(num_lorentz+1)+2])
    y += dy


y = np.gradient(y)
y = y/np.max(np.abs(y))

out = nn.run(y)
print(input)
print(out)

y2 = np.zeros(inputs)
for i in range(num_lorentz):
    dy = lorentz(x,input[i*(num_lorentz+1)+0],out[i],input[i*(num_lorentz+1)+2])
    y2 += dy

y2 = np.gradient(y2)
y = y / np.max(np.abs(y))

plt.plot(x,y)
plt.plot(x,y2,color="black")
plt.title("TRAIN_RPROP")
plt.show()


nn.save("nn_.net")



