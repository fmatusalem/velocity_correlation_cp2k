#!/usr/bin/env python3

#script to calculate velocity autocorrelation function (VAF) from CP2K MD velocities, using OVITO package.

import sys
from ovito.io import import_file
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

program_name = sys.argv[0]
count = len(sys.argv)

if count < 3:
	print('\n\nERROR!!  Enter CP2K velocity trajectory file and MD timestep as arguments!\n\n')
	exit()

fname=sys.argv[1]
dt=float(sys.argv[2])

print('\n\nFile name=', fname)
print('MD Time step', dt)


pipeline = import_file(fname, columns =
         ["Particle Type", "Position.X", "Position.Y", "Position.Z"])

data = pipeline.compute()
nparticles = data.particles.count
nframes = pipeline.source.num_frames

print("Number of frames:",nframes)

# Access the property with the name 'Particle Type':
prop = data.particles.particle_types
ntypes = len(prop.types)       #number of species


nspecies= [1]*ntypes         #number of atoms of each specie
j=0
for i in range(len(prop)-1):
    if prop[i+1] == prop[i]:
        nspecies[j] += 1
        
    else:
        j += 1


# Print list of particle types (their numeric IDs and names)
print("Particle types")
species=['']*ntypes       
for t in prop.types: 
    i=t.id-1
    species[i]=t.name
    print('{} -> {}'.format(t.name, nspecies[i]))


velocities = np.zeros((nparticles, 3, 2*nframes)) #Array to store velocities. Making it double
# the size for the FFT operation, 
# with the second nframes entries equal to zero, to reduce correlations, see Allen & Tildesley


#######power spectrum############
#vibrational density of states
#The default cp2k unit for velocities is bohr/au_time
#convert bohr/au_time to angs/s
#1 bohr = 0.529177 angs
#1 bohr = 5.29177e-11 m
#1 au_time = 2.418884326509*10**-17 s = 0.0242 fs
bohr_to_ang=0.529177/0.0242        #bohr/au_time -> ang/fs
bohr_to_m=5.29177*10**-11/(2.418884326509*10**-17)    #bohr/au_time -> m/s


for frame in range(nframes):
    framedata = pipeline.compute(frame)
    velocities[:,:,frame] = framedata.particles["Position"][:]

velocities = np.multiply(velocities,bohr_to_ang)

v_omega = fft(velocities,axis=2) # FFT along frames for velocities for all particles and all directions
v_omega_star = np.conjugate(v_omega) # Complex conjugate
v_omega_square = np.real(np.multiply(v_omega,v_omega_star)) #Squared modulus


dt = 1.0 # Time step in femtoseconds
ndump = 1 # Number of time steps per dump
nsteps = np.shape(velocities)[2] # Twice number of time steps in file

Delta_t = dt*ndump # Time separation between dumps, in femtoseconds
time_values = Delta_t*np.array(list(range(0,nsteps)))/1000. #In units of ps

#Delta_omega = 2*np.pi/(nsteps*Delta_t)*1000 # delta_omega in units of THz Hz
Delta_omega = 1/(nsteps*Delta_t)*1000 # delta_omega in units of THz Hz
omega_values = Delta_omega*np.array(list(range(0,nsteps))) #In units of THz


#Now average over all particles and directions:
g_omega=[0]*(ntypes+1)


n=0
for i in range(ntypes):
    g_omega[i] = np.average(v_omega_square[n:nspecies[i]+n], axis=0)
    g_omega[i] = np.average(g_omega[i], axis=0)
    n += nspecies[i]

for i in range(ntypes):
	g_omega[ntypes]=g_omega[ntypes]+g_omega[i]


#######################################
#velocity correlation function


acf=[0]*ntypes
n=0
for i in range(ntypes):
    acf[i] = np.real(ifft(g_omega[i]))
    acf[i] = acf[i]/acf[i][0]
    n += nspecies[i]

#saving data to files
print('\n\nVCF and VDOS written to files vcf.dat and vdos.dat!\n\n')

acf=np.array(acf)      #convert list of arrays to arrays of arrays

vcfheader='time ' + ' '.join(species)
np.savetxt('vcf.dat', np.column_stack((time_values, acf.T)), header=str(vcfheader))

g_omega=np.array(g_omega)      #convert list of arrays to arrays of arrays

vdosheader='freq(THz) ' + ' '.join(species) + ' total'
np.savetxt('vdos.dat', np.column_stack((omega_values, g_omega.T)), header=str(vdosheader))

