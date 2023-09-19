from tempfile import NamedTemporaryFile
import numpy as np
import base64 
from scipy.integrate import ode
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

FRAMES = 250
INTERVAL = 100

def set_labels():
  ax.set_zlabel('z')
  ax.set_ylabel('y')
  ax.set_xlabel('x')

def derivative( t, pos_and_vel, q, m, B, E ):
  x, y, z, vx, vy, vz = pos_and_vel[0], pos_and_vel[1], pos_and_vel[2], pos_and_vel[3], pos_and_vel[4], pos_and_vel[5]
  return np.array( [vx, vy, vz, 0, E + q / m * vz * B, - q / m  * vy * B ] )

def simulate( k ):
  idx = int( k * pos_array.shape[0] / FRAMES )
  ax.cla()
  ax.plot3D( pos_array[:idx, 0], 
             pos_array[:idx, 1], 
             pos_array[:idx, 2], "go" )
  # title = "Particle Track under B = " + str(B) + ", E = " + str(E)
  ax.set_title( "Particle Track with No Fields", fontdict = None, loc = 'center', pad = 10 )
  set_labels()

x0 = np.zeros(3)
pos_array = []
v0 = np.zeros(3)
v0[0] = v0[1] = 1
ic = np.concatenate( (x0, v0) )
t0 = 0
t_ = 10
dt = 0.01

B = 0
E = 0

oderes = ode( derivative ).set_integrator('dopri5')
oderes.set_initial_value( ic, t0 ).set_f_params( 1.0, 1.0, B, E ) 

while oderes.successful() and oderes.t < t_:
  oderes.integrate( oderes.t + dt )
  pos_array.append( oderes.y[:3] )

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

pos_array = np.array( pos_array )
res = animation.FuncAnimation( fig, simulate, init_func = set_labels, frames = FRAMES, interval = INTERVAL )
plt.close( res._fig )

# save video
if not hasattr( res, '_encoded_video' ):
  ntf = NamedTemporaryFile( suffix='.mp4', delete = False )
  res.save( ntf.name, fps = 30, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'] )
  ntf.flush()
  output = open( ntf.name, "rb" ).read()
  ntf.close()
  res._encoded_video = base64.b64encode( output ).decode('utf-8')

vt = """<video controls><source src="data:video/x-m4v;base64,{0}" type="video/mp4">Your browser does not support the video tag.</video>"""

HTML( vt.format( res._encoded_video ) )