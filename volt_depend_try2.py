import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import optimize

print("This is the name of the script: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv))
print("The arguments are: ", str(sys.argv))

file_names = sys.argv[1:]
print(file_names)
num_file = len(sys.argv) - 1
print(num_file)

all_currents = []
all_voltages = []

np_gauss_avg = []
ps_gauss_avg = []

def Gauss(x, a, x0, sigma):
	return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

# for loop to process the files.
for k in range(num_file):
	filename_ps = file_names[k]
	#print("Your file is ",filename_ps)
	f_freq = open(filename_ps,'r')
	ps_content = f_freq.readlines()
	f_freq.close()
	
	#print(ps_content[15][18:-1])
	all_currents.append(float(ps_content[15][18:-1]))
	#print(ps_content[17][10:-1])
	all_voltages.append(float(ps_content[17][10:-1]))
	
	# Turn that power spectrum file into an array for python to understand.
	headerlen_ps = 25  # 5 for test file, 25 for real file
	footerlen_ps = 3 # 2 for test file, 2 for real file?
	trim_content_ps = ps_content[headerlen_ps:len(ps_content)-footerlen_ps]
	
	ps_data = []
	for row in range(len(trim_content_ps)):
		temp = trim_content_ps[row].split("\t")
		temp[len(temp)-1]=temp[len(temp)-1].rstrip() # remove new line character (\n) from last element
		for item in range(len(temp)):
			temp[item] =float(temp[item])
		ps_data.append(temp)
	
	freq = ps_data[0]
	ps_data = ps_data[1:]
	
	# Calculate NP
	trace_no = []
	np_data = []
	temp = []
	for row in range(len(ps_data)):
		temp.append(sum(ps_data[row][1:3]))
		temp.append(sum(ps_data[row][3:7]))
		temp.append(sum(ps_data[row][7:15]))
		temp.append(sum(ps_data[row][15:31]))
		temp.append(sum(ps_data[row][31:63]))
		temp.append(sum(ps_data[row][63:127]))
		temp.append(sum(ps_data[row][127:255]))
		np_data.append(temp)
		temp = []
		trace_no.append(row)

	np_data = np.transpose(np_data)

	np_avg = []
	for bin in range(len(np_data)):
		temp = np.mean(np_data[bin])
		np_avg.append(temp)
		np_data[bin] = np_data[bin]
	# print(np_avg)
	# print(len(np_data))
	
	m = np_data
	# print(m)

	# copying the terms from 'sort_NP_by_bin.np' until creativity strikes.

	bin_boundaries = []
	bin_bounds = []
	bin_centers = []
	for bin in range(len(m)):
		minimum = np.log10(min(m[bin]))
		#print(minimum)
		maximum = np.log10(max(m[bin]))
		#print(maximum)
		delta = (maximum - minimum)/50
		#print(delta)
		bin_bounds.append(np.linspace(minimum,maximum,51))
		bin_centers.append(np.linspace(minimum + delta, maximum - delta, 50))
	#bin_bounds = np.linspace(0.0, 2.5,51)
	#bin_centers = np.linspace(0.025,2.475,50)
	#print(bin_bounds)
	#print(bin_centers)
	counter = []
	bin_boundaries = []
	for bin in range(len(m)):
		temp1, temp2 = np.histogram(np.log10(m[bin]),bins=bin_bounds[bin])
		counter.append(temp1)
		bin_boundaries.append(bin_centers[bin])
	#print(len(counter[0]))
	#print(len(bin_boundaries[0]))
	
	temp = []
	
	for i in range(len(bin_boundaries)):
		x = np.array(bin_boundaries[i])
		y = np.array(counter[i])
		
		#print(x)
		#print(y)
		
		mean = sum(x*y)/sum(y)
		#print(10**mean)
		sigma = np.sqrt(sum(y*(x-mean)**2)/ sum(y))
		#print(sigma)
		#print(max(y))
		
		popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])
		temp.append(10**(popt[1]))
		#print("The average from your graph is: " +str( 10**(popt[1])))
		#print("The low FWHM value is:" +str(10**(popt[1]-np.sqrt(2*np.log(2))*popt[2])))
		#print("The high FWHM value is:" +str(10**(popt[1]+np.sqrt(2*np.log(2))*popt[2])))

	np_gauss_avg.append(temp)
	temp = []
	
	ps_data = np.transpose(ps_data)
	
	m = []
	m.append(ps_data[3]) # freq = 10 Hz
	m.append(ps_data[19]) # freq = 50 Hz
	m.append(ps_data[39]) # freq = 100 Hz
	m.append(ps_data[79]) # freq = 200 Hz
	m.append(ps_data[199]) # freq = 500 Hz
	
	#print(len(m))
	
	# copying the terms from 'sort_NP_by_bin.np' until creativity strikes.

	bin_boundaries = []
	bin_bounds = []
	bin_centers = []
	for bin in range(len(m)):
		minimum = np.log10(min(m[bin]))
		#print(minimum)
		maximum = np.log10(max(m[bin]))
		#print(maximum)
		delta = (maximum - minimum)/50
		#print(delta)
		bin_bounds.append(np.linspace(minimum,maximum,51))
		bin_centers.append(np.linspace(minimum + delta, maximum - delta, 50))
	#bin_bounds = np.linspace(0.0, 2.5,51)
	#bin_centers = np.linspace(0.025,2.475,50)
	#print(bin_bounds)
	#print(bin_centers)
	counter = []
	bin_boundaries = []
	for bin in range(len(m)):
		temp1, temp2 = np.histogram(np.log10(m[bin]),bins=bin_bounds[bin])
		counter.append(temp1)
		bin_boundaries.append(bin_centers[bin])
	#print(len(counter[0]))
	#print(len(bin_boundaries[0]))
	
	temp = []
	
	for i in range(len(bin_boundaries)):
		x = np.array(bin_boundaries[i])
		y = np.array(counter[i])
		
		#print(x)
		#print(y)
		
		mean = sum(x*y)/sum(y)
		#print(10**mean)
		sigma = np.sqrt(sum(y*(x-mean)**2)/ sum(y))
		#print(sigma)
		#print(max(y))
		
		popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])
		temp.append(10**(popt[1]))
		#print("The average from your graph is: " +str( 10**(popt[1])))
		#print("The low FWHM value is:" +str(10**(popt[1]-np.sqrt(2*np.log(2))*popt[2])))
		#print("The high FWHM value is:" +str(10**(popt[1]+np.sqrt(2*np.log(2))*popt[2])))

	ps_gauss_avg.append(temp)
	temp = []
	
print(all_currents)
print(all_voltages)
np_gauss_avg = np.transpose(np_gauss_avg)
print(np_gauss_avg)
ps_gauss_avg = np.transpose(ps_gauss_avg)
print(ps_gauss_avg)

# Now to make all 4 sets of the plots to print.

all_voltages = np.log10(all_voltages)
all_currents = np.log10(all_currents)
np_gauss_avg = np.log10(np_gauss_avg)
ps_gauss_avg = np.log10(ps_gauss_avg)

freq = [10,50, 100, 200, 500]
slopes1 = []
intercept1 = []
for i in range(len(ps_gauss_avg)):
	fitfunc = lambda p, x: p[0] + p[1] * x
	errfunc = lambda p, x, y: (y - fitfunc(p, x))
	pinit = [1.0, -1.0]
	out = optimize.leastsq(errfunc, pinit, args=(all_currents, ps_gauss_avg[i]))
	#print(out)
	pfinal = out[0]
	#print(pfinal)
	slopes1.append(pfinal[1])
	intercept1.append(pfinal[0])
	

print(slopes1)
print(intercept1)

curr_pos = 0

def key_event1(e1):
    global curr_pos

    if e1.key == "right":
        curr_pos = curr_pos + 1
    elif e1.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % len(ps_gauss_avg)

    ax1.cla()
    ax1.plot(all_currents, ps_gauss_avg[curr_pos],'ko')
    ax1.plot(all_currents, intercept1[curr_pos]+slopes1[curr_pos]*all_currents, 'b-')
    ax1.set_title('Current Dependence of Noise Power Spectrum Magnitude \n at Frequency '+str(freq[curr_pos])+' Hz')
    ax1.set_xlabel('log(Current)')
    ax1.set_ylabel('log(Noise Power Specture Magnitude)')
    ax1.minorticks_on()
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    fig1.canvas.draw()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
fig1.canvas.mpl_connect('key_press_event', key_event1)
# Initialize by plotting the first plot.
ax1.plot(all_currents, ps_gauss_avg[curr_pos],'ko')
ax1.plot(all_currents, intercept1[curr_pos]+slopes1[curr_pos]*all_currents, 'b-')
ax1.set_title('Current Dependence of Noise Power Spectrum Magnitude \n at Frequency '+str(freq[curr_pos])+' Hz')
ax1.set_xlabel('log(Current)')
ax1.set_ylabel('log(Noise Power Spectrum Magnitude)')
ax1.minorticks_on()
ax1.grid(which='major', linestyle='-', linewidth='0.5', color='black')
ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')	
		
plt.figure()
plt.semilogx(freq, slopes1,'ko')
# Don't allow the axis to be on top of your data
#plt.set_axisbelow(True)
plt.title('Current Dependence Slope b vs Frequency \n (currents, individual frequencies)')
plt.xlabel('log(Frequency [Hz])')
plt.ylabel('Spectral Slope b')
# Turn on the minor TICKS, which are required for the minor GRID
plt.minorticks_on()
# Customize the major grid
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')		
slopes2 = []
intercept2 = []
for i in range(len(ps_gauss_avg)):
	fitfunc = lambda p, x: p[0] + p[1] * x
	errfunc = lambda p, x, y: (y - fitfunc(p, x))
	pinit = [1.0, -1.0]
	out = optimize.leastsq(errfunc, pinit, args=(all_voltages, ps_gauss_avg[i]))
	#print(out)
	pfinal = out[0]
	#print(pfinal)
	slopes2.append(pfinal[1])
	intercept2.append(pfinal[0])

print(slopes2)
print(intercept2)

curr_pos = 0

def key_event2(e2):
    global curr_pos

    if e2.key == "right":
        curr_pos = curr_pos + 1
    elif e2.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % len(ps_gauss_avg)

    ax2.cla()
    ax2.plot(all_voltages, ps_gauss_avg[curr_pos],'ko')
    ax2.plot(all_voltages, intercept2[curr_pos]+slopes2[curr_pos]*all_voltages, 'b-')
    ax2.set_title('Voltage Dependence of Noise Power Spectrum Magnitude \n at Frequency '+str(freq[curr_pos])+' Hz')
    ax2.set_xlabel('log(Voltage)')
    ax2.set_ylabel('log(Noise Power Spectrum Magnitude)')
    ax2.minorticks_on()
    ax2.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    fig2.canvas.draw()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
fig2.canvas.mpl_connect('key_press_event', key_event2)
# Initialize by plotting the first plot.
ax2.plot(all_voltages, ps_gauss_avg[curr_pos],'ko')
ax2.plot(all_voltages, intercept2[curr_pos]+slopes2[curr_pos]*all_voltages, 'b-')
ax2.set_title('Voltage Dependence of Noise Power Spectrum Magnitude \n at Frequency '+str(freq[curr_pos])+' Hz')
ax2.set_xlabel('log(Voltage)')
ax2.set_ylabel('log(Noise Power Spectrum Magnitude)')
ax2.minorticks_on()
ax2.grid(which='major', linestyle='-', linewidth='0.5', color='black')
ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='black')	
	
plt.figure()
plt.semilogx(freq, slopes2,'ko')
# Don't allow the axis to be on top of your data
#plt.set_axisbelow(True)
plt.title('Voltage Dependence Slope b vs Frequency \n (voltages, individual frequencies)')
plt.xlabel('log(Frequency [Hz])')
plt.ylabel('Spectral Slope b')
# Turn on the minor TICKS, which are required for the minor GRID
plt.minorticks_on()
# Customize the major grid
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')	

freq = [5.728, 12.809, 26.955, 55.241, 111.810, 224.948, 451.222]
slopes3 = []
intercept3 = []
for i in range(len(np_gauss_avg)):
	fitfunc = lambda p, x: p[0] + p[1] * x
	errfunc = lambda p, x, y: (y - fitfunc(p, x))
	pinit = [1.0, -1.0]
	out = optimize.leastsq(errfunc, pinit, args=(all_currents, np_gauss_avg[i]))
	#print(out)
	pfinal = out[0]
	#print(pfinal)
	slopes3.append(pfinal[1])
	intercept3.append(pfinal[0])
	
print(slopes3)
print(intercept3)

curr_pos = 0

def key_event3(e3):
    global curr_pos

    if e3.key == "right":
        curr_pos = curr_pos + 1
    elif e3.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % len(np_gauss_avg)

    ax3.cla()
    ax3.plot(all_currents, np_gauss_avg[curr_pos],'ko')
    ax3.plot(all_currents, intercept3[curr_pos]+slopes3[curr_pos]*all_currents, 'b-')
    ax3.set_title('Current Dependence of Noise Power Magnitude \n in Bin '+str(curr_pos +1))
    ax3.set_xlabel('log(Current)')
    ax3.set_ylabel('log(Noise Power Magnitude)')
    ax3.minorticks_on()
    ax3.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax3.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    fig3.canvas.draw()

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
fig3.canvas.mpl_connect('key_press_event', key_event3)
# Initialize by plotting the first plot.
ax3.plot(all_currents, np_gauss_avg[curr_pos],'ko')
ax3.plot(all_currents, intercept3[curr_pos]+slopes3[curr_pos]*all_currents, 'b-')
ax3.set_title('Current Dependence of Noise Power Magnitude \n in Bin '+str(curr_pos +1)+' Hz')
ax3.set_xlabel('log(Current)')
ax3.set_ylabel('log(Noise Power Magnitude)')
ax3.minorticks_on()
ax3.grid(which='major', linestyle='-', linewidth='0.5', color='black')
ax3.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.figure()
plt.semilogx(freq, slopes3,'ko')
# Don't allow the axis to be on top of your data
#plt.set_axisbelow(True)
plt.title('Current Dependence Slope b vs Frequency \n (currents, NP bins)')
plt.xlabel('log(Frequency [Hz])')
plt.ylabel('Spectral Slope b')
# Turn on the minor TICKS, which are required for the minor GRID
plt.minorticks_on()
# Customize the major grid
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')		
slopes4 = []
intercept4 = []
for i in range(len(np_gauss_avg)):
	fitfunc = lambda p, x: p[0] + p[1] * x
	errfunc = lambda p, x, y: (y - fitfunc(p, x))
	pinit = [1.0, -1.0]
	out = optimize.leastsq(errfunc, pinit, args=(all_voltages, np_gauss_avg[i]))
	#print(out)
	pfinal = out[0]
	#print(pfinal)
	slopes4.append(pfinal[1])
	intercept4.append(pfinal[0])

print(slopes4)
print(intercept4)

curr_pos = 0

def key_event4(e4):
    global curr_pos

    if e4.key == "right":
        curr_pos = curr_pos + 1
    elif e4.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % len(np_gauss_avg)

    ax4.cla()
    ax4.plot(all_voltages, np_gauss_avg[curr_pos],'ko')
    ax4.plot(all_voltages, intercept4[curr_pos]+slopes4[curr_pos]*all_voltages, 'b-')
    ax4.set_title('Voltage Dependence of Noise Power Magnitude \n in Bin '+str(curr_pos +1)+' Hz')
    ax4.set_xlabel('log(Voltage)')
    ax4.set_ylabel('log(Noise Power Magnitude)')
    ax4.minorticks_on()
    ax4.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax4.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    fig4.canvas.draw()

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
fig4.canvas.mpl_connect('key_press_event', key_event4)
# Initialize by plotting the first plot.
ax4.plot(all_voltages, np_gauss_avg[curr_pos],'ko')
ax4.plot(all_voltages, intercept4[curr_pos]+slopes4[curr_pos]*all_voltages, 'b-')
ax4.set_title('Voltage Dependence of Noise Power Magnitude \n in Bin '+str(curr_pos +1)+' Hz')
ax4.set_xlabel('log(Voltage)')
ax4.set_ylabel('log(Noise Power Magnitude)')
ax4.minorticks_on()
ax4.grid(which='major', linestyle='-', linewidth='0.5', color='black')
ax4.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.figure()
plt.semilogx(freq, slopes4,'ko')
# Don't allow the axis to be on top of your data
#plt.set_axisbelow(True)
plt.title('Voltage Dependence Slope b vs Frequency \n (voltages, NP bins)')
plt.xlabel('log(Frequency [Hz])')
plt.ylabel('Spectral Slope b')
# Turn on the minor TICKS, which are required for the minor GRID
plt.minorticks_on()
# Customize the major grid
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')	



plt.show()