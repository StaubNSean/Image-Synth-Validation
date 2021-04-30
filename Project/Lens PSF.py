  
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:03:43 2021
@author: Sean
Prysm author: Brandon Dube
"""

from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.geometry import circle
from prysm.polynomials import hopkins
from prysm.propagation import Wavefront

def circular_lens_spherical_aberration(d, fno, wlen, samp, Q, z):
    """Constructs the Intensity/Incoherent point spread function of a
    circular lens, accounting for aperture diffraction and spherical
    aberration.
    
    This function is used to construct what is called the "intensity" or 
    "incoherent" point spread function (PSF) of a circular lens. Aperture diffraction
    and spherical aberration are accounted for. This function allows the user
    to specify the transverse diamter 'd' and the F-Number 'fno' of the lens.
    The PSF of a lens is wavelength dependent and the user specifies this using
    'wlen'. Since this is a digital simulation of a lens, the user needs to
    specify how many samples will be used to model the digital plane where
    the lens exists, this is done with 'samp'. The user also needs to define
    'Q', which can be thought of as a padding factor that helps improve 
    spatial resolution of the PSF. The spherical aberration modeled here
    requires the user to scale the "magnitude" of the effect the 
    spherical aberration will have on the incoming wavefront, this is done
    by specifying 'z'.
    
    Parameters
    ----------
    d : 'int'
        Diamter of lens, units of mm, millimeters.    
    fno : 'int'
        F-Number of lens, units mm/mm, millimeters/millimeters, no units.
    wlen : 'float'
        Wavelength of light to be used in forward model, units of um, micrometers. 
    samp : 'int'
        Number of samples, "pixels", to be used in forward model, 
        will be broadcast to x and y dimensions.
    Q : 'int'
        Padding factor used when constructing field intensity in focal/PSF plane,
        improves spatial resolution.
    z : 'int'
        Zero-to-peak spherical aberration value, units on nm, nanometers.
    Returns
    -------
    psf : RichData object
        Intensity/Incoherent point spread function of lens.
        
    Examples
    --------
    >>> PSF = circular_lens_spherical_aberration(10, 10, 0.6328, 256, 8, 100)
    >>>
    """
    # Construct sample grid with side lengths d, xi and eta are cartesian components.
    xi, eta = make_xy_grid(samp, diameter=d)
    # Convert grid to polar coordinates.
    r, theta = cart_to_polar(xi, eta)
    
    # Construct amplitude function, functions as limiting aperture and aberration amplitude.
    A = circle(d/2, r)
    
    # Construct radial polar coordinate grid scaled to max value of 1,
    # necessary when using Hopkins polynomials. To be used in construction of phase function.
    rho = r / d/2
    # Construct phase function modeling spherical aberration using Hopkins polynomials.
    # zero-to-peak value set to 1, can/will be scaled to desired value.
    phi = hopkins(0, 4, 0, rho, theta, 1)
    # Scale zero-to-peak value of spherical aberration to desired amount using input z.
    phi = phi * z * d/2
    
    # Use cartesian grid array to deduce inter-sample ("pixel") spacing.
    dx = xi[0,1]-xi[0,0]
    
    # Construct pupil function from amplitude/phase function, wavelength, and inter-sample spacing.
    P = Wavefront.from_amp_and_phase(A, phi, wlen, dx)
    
    # Construct field intensity in focal plane,
    # .focus is propagating the wavefront from aperture/pupil plane to focal/PSF plane
    E = P.focus(d*fno, Q=Q)
    # Construct Intensity/Incoherent PSF from field intensity.
    # .intensity is taking the Fourier transform of the field intensity.
    psf = E.intensity
    
    return psf

def plot_PSF(psf,wlen, fno, clim=None):
    """Plots the Intensity/Incoherent point spread function of a lens.
    
    This function plots a given PSF. The PSF of a lens can
    be thought of as a filter that can be applied to an
    image by convolution to simulate the imaging capabilites of
    a given lens. The PSF is generally quite small and this needs to
    be accounted for when plotting. Specifying the wavelength, 'wlen', and 
    F-Number, 'fno', allow the function to automatically scale the range
    in which to plot the PSF. The input variable 'clim' is the contrast
    limit to be used, and can be selected to help visualize the PSF.
    Parameters
    ----------
    psf : RichData object
        Intensity/Incoherent point spread of function of lens to be plotted.  
    wlen : 'float'
        Wavelength of light to be used in forward model, units of um, micrometers,
        should match wavelength used to construct psf.
    fno : 'int'
        F-Number of lens, units mm/mm, millimeters/millimeters, no units in reality,
        should match F-number used to construct psf.
    clim : 'tuple' of floats, optional
        Contrast limits, used to improve interpretability of psf plot. The default is None.
    Returns
    -------
    None.
    
    Examples
    --------
    >>> plot_PSF(psf, 0.6328, 10, (1e5,3e9))
    >>> "Generates a plot"
    
    """
    # Constructs plotting region to be displayed.
    psf_radius = 1.22*wlen*fno
    # Plots Intensity/Incoherent PSF.
    psf.plot2d(xlim=psf_radius*5, log=True, cmap='gray',
           clim=clim, interpolation='bilinear')
