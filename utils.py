import numpy as np

# %%
# Define useful transforms for when we're reading the data in
def Get_PtEtaPhiM_fromXYZT(obj_px, obj_py, obj_pz, obj_e):
    '''
    Takes in arrays of shape (n_batch[,n_obj],) for x, y, z and t (==e) 
    of some objects and returns arrays of shape (n_batch[,n_obj],) containing 
    the Pt, Eta, Phi and M of the objects. Each element of n_batch corresponds
    to one event, and each of the n_objs represents an object in the event.
    '''
    obj_pt = np.sqrt((obj_px ** 2 + obj_py**2))
    obj_ptot = np.sqrt((obj_px ** 2 + obj_py**2 + obj_pz**2)) # sqrt*(x^2 + y^2 + z^2) == rho in spherical coords
    
    obj_cosTheta = np.empty_like(obj_px)
    obj_cosTheta[obj_ptot==0] = 1
    obj_cosTheta[obj_ptot!=0] = obj_pz[obj_ptot!=0]/obj_ptot[obj_ptot!=0]

    obj_Eta = np.empty_like(obj_px)
    eta_valid_mask = obj_cosTheta*obj_cosTheta < 1
    obj_Eta[eta_valid_mask] = -0.5* np.log( (1.0-obj_cosTheta[eta_valid_mask])/(1.0+obj_cosTheta[eta_valid_mask]) )
    obj_Eta[(~eta_valid_mask) & (obj_pz==0)] = 0
    obj_Eta[(~eta_valid_mask) & (obj_pz>0)] = np.inf
    obj_Eta[(~eta_valid_mask) & (obj_pz<0)] = -np.inf
    
    obj_Phi = np.empty_like(obj_px)
    phi_valid_mask = ~((obj_px == 0) & (obj_py == 0))
    obj_Phi[phi_valid_mask] = np.atan2(obj_py[phi_valid_mask], obj_px[phi_valid_mask])
    obj_Phi[~phi_valid_mask] = 0

    obj_Mag2 = obj_e**2 - obj_ptot**2

    obj_M = np.empty_like(obj_px)
    obj_M[obj_Mag2<0] = -np.sqrt((-obj_Mag2[obj_Mag2<0]))
    obj_M[obj_Mag2>=0] = np.sqrt((obj_Mag2[obj_Mag2>=0]))

    return obj_pt, obj_Eta, obj_Phi, obj_M

# %% Temporary testing cell
def GetXYZT_FromPtEtaPhiM(pt, eta, phi, m):
    '''
    Takes in arrays of shape (n_batch[,n_obj],) for Pt, Eta, Phi and M
    of some objects and returns arrays of shape (n_batch[,n_obj],) containing 
    the X, Y, Z, and T(==E) of the objects. Each element of n_batch corresponds
    to one event, and each of the n_objs represents an object in the event.
    '''
    x = pt*np.cos(phi)
    y = pt*np.sin(phi)
    z = pt*np.sinh(eta)
    t = (np.sqrt(x*x + y*y + z*z + m*m))*(m >= 0) + (np.sqrt(np.maximum((x*x + y*y + z*z - m*m), np.zeros(len(m)))))*(m < 0)
    return x, y, z, t


def GetXYZT_FromPtEtaPhiE(pt, eta, phi, E):
    x,y,z,_=GetXYZT_FromPtEtaPhiM(pt, eta, phi, np.zeros_like(E))
    return x, y, z, E

def Rotate4VectorPhi(x, # (n_batch[,n_obj],)
                  y, # (n_batch[,n_obj],)
                  z, # (n_batch[,n_obj],)
                  t, # (n_batch[,n_obj],)
                  phi, # (n_batch[,n_obj],)
                   ):
    newx = x*np.cos(phi) + y*np.sin(phi)
    newy = -x*np.sin(phi) + y*np.cos(phi)
    return newx, newy, z, t

def Rotate4VectorEta(x, # (n_batch[,n_obj],)
                  y, # (n_batch[,n_obj],)
                  z, # (n_batch[,n_obj],)
                  t, # (n_batch[,n_obj],)
                  eta, # (n_batch[,n_obj],)
                   ):
    theta = 2*np.arctan(np.exp(-eta))
    newx = x*np.cos(theta) + z*np.sin(theta)
    newy = y*np.cos(theta) + z*np.sin(theta)
    newz = -x*np.sin(theta) + y*np.cos(theta)
    return newx, newy, newz, t


def Rotate4VectorPhiEta(x, # (n_batch[,n_obj],)
                  y, # (n_batch[,n_obj],)
                  z, # (n_batch[,n_obj],)
                  t, # (n_batch[,n_obj],)
                  phi, # (n_batch[,n_obj],)
                  eta, # (n_batch[,n_obj],)
                   ):
    # print(x, y, z, t)
    newx, newy, _, _ = Rotate4VectorPhi(x,y,z,t,phi)
    # print(newx, newy, z, t)
    # assert((newy<1e-3).all()) # Only if we're rotating by the angle of the jet itself!
    theta = np.pi/2 - 2*np.arctan(np.exp(-eta))
    # print(theta)
    # print(np.arctan(z/newx))
    # print(np.arctan(newx/z))
    # newx = newx*np.cos(np.pi/2 - theta) + z*np.sin(np.pi/2 - theta) # IT WORKED KINDA WITH THIS
    # newz = -newx*np.sin(np.pi/2 - theta) + z*np.cos(np.pi/2 - theta) # IT WORKED KINDA WITH THIS
    newnewx = newx*np.cos(theta) + z*np.sin(theta)
    newz = -newx*np.sin(theta) + z*np.cos(theta)
    return newnewx, newy, newz, t
