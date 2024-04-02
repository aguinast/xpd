import sys, os
from PyQt5.QtCore import (Qt, pyqtSignal, QRect, QEvent)
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox, QTabWidget,QMainWindow, QMenu, QPushButton, QRadioButton, QVBoxLayout, 
                                QHBoxLayout, QWidget, QSlider, QLabel, QLineEdit, QColorDialog, QCheckBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5 import ToolbarQt
from matplotlib.backend_managers import ToolManager
from matplotlib import backend_tools
from matplotlib.patches import Arrow

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['toolbar'] = 'toolmanager'
from matplotlib.backend_tools import ToolBase, ToolToggleBase
import numpy as np
import xrayutilities as xu
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

version = 'en'
names = {'br':{'title':'Padr\u00e3o de Difra\u00e7\u00e3o de Raios X',
                'pars': u'Par\u00e2metros de Rede (\u212b)', 
                'wvl': 'Comprimento de Onda (\u212b) e Energia (keV)',
                'size': 'Tamanho Te\u00f3rico de Cristalito (\u212b)', 
                'hkl': '\u00cdndices de Miller (hkl)', 
                'int': 'Intensidade M\u00e1xima', 
                'atm': '\u00c1tomo', 
                'crystal': 'C\u00e9lula Unit\u00e1ria', 
                'base': '\u00c1tomos da Base', 
                'label_0': '\u00c1tomo da posi\u00e7\u00e3o (x,y,z) = (0,0,0)',
                'info': 'Infos', 
                'welcome': 'Bem vindo ao XPD sim', 
                'xpd':'Padr\u00e3o de Difra\u00e7\u00e3o de Raios X', 
                'lattpar': 'Par\u00e2metros da Rede',
                'a': 'Par\u00e2metro "a"',  'b': 'Par\u00e2metro "b"', 'c': 'Par\u00e2metro "c"', 
                'alpha': '\u00e2ngulo entre os eixos definidos por "b" e "c"', 'beta': '\u00e2ngulo entre os eixos definidos por "a" e "c"', 'gamma': '\u00e2ngulo entre os eixos definidos por "a" e "b"', 
                'complement_0': 'em \u212b', 
                'complement_1': 'em graus', 
                'problem_a': 'n\u00e3o \u00e9 um valor v\u00e1lido \npara',
                'en_0': 'energia', 'wvl_0': 'comprimento de onda', 'size_0': 'tamanho de cristalito', 'atm_0': 'posi\u00e7\u00e3o do \u00e1tomo',
                'freeze0':'congelar curva', 'freeze1':r'cor {}', 'red': 'vermelha', 'green':'verde', 'blue':'azul',
                'rescale':'Ajustar a escala da intensidade',
                'tt_b_pars':'Clique para voltar ao valor inicial', 'tt_s_pars':'Clique e arraste para mudar','tt_e_pars':'Entre um valor',
                'color_0':'Mudar a cor deste \u00e1tomo',
                'crysedge': 'Mostrar/esconder aresta', 'crysface': 'Mostrar/esconder face do cristal', 'crysatoms': 'Mostrar/esconder \u00e1tomos',
                'hkl_000': 'hkl deve ser diferente de "0 0 0"', 'hkl_inf': 'entre um HKL para seguir um pico', 'hkl_inf2': '(valor m\u00e1ximo plotado: 6)',
                'showHKL':'clique para mostrar/ocultar HKL',
                'tt_s_uc': 'Clique e arraste para dar "zoom" no cristal', 
                'addAtoms': 'Adicionar um \u00e1tomo na base', 'remAtoms': 'Remover o \u00faltimo \u00e1tomo da base',
                'posx': 'posi\u00e7\u00e3o x', 'posy': 'posi\u00e7\u00e3o y', 'posz': 'posi\u00e7\u00e3o z', 
                'atomtype': 'Tipo do \u00e1tomo: entre um novo para trocar',
                },
         'en':{'title':'X-ray Difraction Pattern', 'pars': u'Lattice Parameters (\u212b)',     'wvl': 'Wavelength (\u212b) and Energy (keV)',
                'size': 'Theoretical Crystallite Size (\u212b)',  'hkl': 'Miller Indexes (hkl)',    'int': 'Máximum Intensity', 
                'atm': 'Atom', 'crystal': 'Unit Cell' , 'base': 'Base Atoms', 'label_0': 'Atom of position (x,y,z) = (0,0,0)',
                'info': 'Infos', 'welcome': 'Welcome to XPD sim', 'xpd':'X-ray Powder Diffraction Pattern', 'lattpar':'Lattice Parameters',
                'a': 'Lattice Parameter "a"',  'b': 'Lattice Parameter "b"', 'c': 'Lattice Parameter "c"', 'alpha': 'Angle between axes defined by "b" e "c"', 
                'beta': 'Angle between axes defined by "a" e "c"', 'gamma': 'Angle between axes defined by "a" e "b"',
                'complement_0': 'in \u212b', 'complement_1': 'in degrees', 'problem_a': 'is not a valid value \nfor',
                'en_0': 'energy', 'wvl_0': 'wavelength', 'size_0': 'crystal size', 'atm_0': 'atom position',
                'freeze0':'freeze curve', 'freeze1':r'{} color', 'red': 'red', 'green':'green', 'blue':'blue',
                'rescale':'Intensity rescale',
                'tt_b_pars':'Click to reset value', 'tt_s_pars':'Click and drag to change', 'tt_e_pars':'Enter value',
                'color_0':'Change this atom color',
                'crysedge': 'Show/hide edge', 'hkl_000': 'hkl should be different from "0 0 0"', 'hkl_inf': 'enter an HKL to follow a peak', 'hkl_inf2': '(maximum plotted value: 6)',
                'showHKL':'click to show/hide HKL',
                'tt_s_uc': 'Click and drag to zoom crystal', 'crysface': 'Show/hide crystal faces', 'crysatoms': 'Show/hide atoms',
                'addAtoms': 'Add one atom into the base', 'remAtoms': 'Remove last atom from base',
                'posx': 'x position', 'posy': 'y position', 'posz': 'z position', 
                'atomtype': 'Atom type: enter a new one to change',
                },
         'es':{'title':'Patr\u00f3n de Difracci\u00f3n de Rayos X', 'pars': u'Par\u00e1metros de Red (\u212b)', 'wvl': 'Longitud de Onda (\u212b) y Energ\u00eda (keV)',
                'size': 'Tama\u00f1o Te\u00f3rico de Cristalito (\u212b)',  'hkl': '\u00cdndices de Miller (hkl)', 'int': 'Intensidad M\u00e1xima', 
                'atm': '\u00c1tomo', 'crystal': 'Celda Unitaria', 'base': '\u00c1tomos de la Base', 'label_0': '\u00c1tomo de la posici\u00f3n (x,y,z) = (0,0,0)',
                'info': 'Informaciones', 'welcome': 'Bien venido al XPD sim', 'xpd':'Patr\u00f3n de Difracci\u00f3n de Rayos X', 'lattpar': 'Par\u00e1metros de Red',
                'a': 'Par\u00e1metro "a"',  'b': 'Par\u00e1metro "b"', 'c': 'Par\u00e1metro "c"', 'alpha': '\u00c1ngulo entre ejes definidos por "b" y "c"',
                'beta': '\u00c1ngulo entre ejes definidos por "a" y "c"', 'gamma': '\u00c1ngulo entre ejes definidos por "a" y "b"', 
                'complement_0': 'en \u212b', 'complement_1': 'en grados', 'problem_a': 'no es un valor v\u00e1lido \npara',
                'en_0': 'energ\u00eda', 'wvl_0': 'longitud de onda', 'size_0': 'tama\u00f1o de cristalito', 'atm_0': 'posici\u00f3n del \u00e1tomo',
                'freeze0':'congelar curva', 'freeze1':r'color {}', 'red': 'roja', 'green':'verde', 'blue':'azul',
                'rescale':'Ajustar la escala de la intensidad',
                'tt_b_pars':'Haga clic para volver al valor inicial', 'tt_s_pars':'Haga clic y arrastre para cambiar', 'tt_e_pars':'Introduce un valor',
                'color_0':'Cambiar color del \u00e1tomo',
                'crysedge': 'Mostrar/ocultar borde', 'hkl_000': 'hkl deve ser diferente de "0 0 0"', 'hkl_inf': 'ingrese un HKL para seguir un pico', 'hkl_inf2': '(valor m\u00e1ximo trazado: 6)',
                'showHKL':'haga clic para mostrar/ocultar HKL',
                'tt_s_uc': 'Haga clic y arrastre para acercar o alejar el cristal', 'crysface': 'Mostrar/ocultar faces', 'crysatoms': 'Mostrar/ocultar \u00e1tomos',
                'addAtoms': 'A\u00f1adir un \u00e1tomo en la base', 'remAtoms': 'Quitar el \u00faltimo \u00e1tomo de la base',
                'posx': 'posici\u00f3n x', 'posy': 'posici\u00f3n y', 'posz': 'posici\u00f3n z', 
                'atomtype': 'Tipo de \u00e1tomo: ingrese uno nuevo para cambiar',
                }}
                
font = 'Currier'
fontsize = 11

class DoubleSlider(QSlider):

    # create our signal that we can connect to if necessary
    doubleValueChanged = pyqtSignal(float)

    def __init__(self, decimals=3, *args, **kargs):
        super(DoubleSlider, self).__init__( *args, **kargs)
        self._multi = 10 ** decimals

        self.valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        value = float(super(DoubleSlider, self).value())/self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super(DoubleSlider, self).value()) / self._multi

    def setMinimum(self, value):
        return super(DoubleSlider, self).setMinimum(int(value * self._multi))

    def setMaximum(self, value):
        return super(DoubleSlider, self).setMaximum(int(value * self._multi))

    def setSingleStep(self, value):
        return super(DoubleSlider, self).setSingleStep(int(value * self._multi))

    def singleStep(self):
        return float(super(DoubleSlider, self).singleStep()) / self._multi

    def setValue(self, value):
        super(DoubleSlider, self).setValue(int(value * self._multi))

class Icons():
    def __init__(self):
        import matplotlib
        fig, ax = plt.subplots()
        n = 30.
        w = 5
        x = np.arange(n)
        y0 = np.exp((x-n/2.-n/15.)*(n/2.+n/15.-x)/(2.*n*n/100.))
        y1 = np.exp((x-n/2.+n/15.)*(n/2.-n/15.-x)/(2.*n*n/100.))
        
        line0, = ax.plot(x,y0, c= 'k', lw = w)
        line1, = ax.plot(x,y0, c= 'k', lw = w)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_ylim(-0.2*y0.max()+y0.min(),y0.max()*1.4)
        for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(w*2/3.)
        self.name = []
        for j,i in enumerate(['red', 'green', 'blue']):
            line1.set_ydata(y1)
            line1.set_color(i)
            fig.set_size_inches(1,1)
            self.name.append(matplotlib.get_data_path() + r"\images\freeze_" + '{}'.format(i))
            plt.savefig(self.name[j] + '_large.png', dpi=48)
            plt.savefig(self.name[j] + '.png', dpi=24)
        y2 = 0.5*np.exp((x-n/2.)*(n/2.-x)/(2.*n*n/100.))
        y3 = 0.3+0.8*np.exp((x-n/2.)*(n/2.-x)/(2.*n*n/100.))
        line0.set_ydata(y3)
        line1.set_ydata(y2)
        line1.set_color('#aaaaaa')
        name = matplotlib.get_data_path() + r"\images\rescale"
        self.name.append(name)
        plt.savefig(name + '_large.png', dpi=48)
        plt.savefig(name + '.png', dpi=24)
        fig.clf()

class Freeze(ToolToggleBase):
    """Freeze simulation."""
    default_toggled = False
    def __init__(self, *args, curve, plot, main_plot, **kwargs):
        self.curve_dict = {0:'r', 1:'g', 2:'b'}
        self.curve_color = {0:'red', 1:'green', 2:'blue'}
        self.plot = plot
        self.main_plot = main_plot
        super().__init__(*args, **kwargs)
        self.default_keymap = self.curve_dict[curve]
        self.description = names[version]['freeze0'] + ' (' + names[version]['freeze1'].format(names[version][self.curve_color[curve]]) + ')'
        self.image = a.name[curve]
    def enable(self, *args):
        self.set_freeze(True)
    def disable(self, *args):
        self.set_freeze(False)
    def set_freeze(self, state):
        if state:
            self.plot.set_ydata(self.main_plot.get_ydata())
        self.plot.set_visible(state)
        self.figure.canvas.draw()

class Rescale_y(ToolBase):
    """Rescale the XPD figure, showing all data."""
    default_keymap = 'y'
    description = names[version]['rescale']
    
    #r'C:\exec\python\not_tested\freeze'
    
    def __init__(self, *args, func, main_plot, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = a.name[3]
        self.func = func
        self.main_plot = main_plot
        #print (self.image)
    def trigger(self, *args, **kwargs):
        self.func()
        self.figure.canvas.draw()

class Window(QMainWindow,QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.initial_parameters()

        self.setGeometry(10, 30, self.window_w, self.window_h)

        self.include_XPDFigure().setGeometry(*self.XPD_Figure_geo())
        
        ## this is just to include a possibility to compare some x, y pxrd datafile, trying to remove the first lines
        try:
            if os.path.isfile(sys.argv[1]):
                for i in range (10):
                    try:
                        x,y = np.transpose(np.loadtxt(sys.argv[1], skiprows=i, usecols=(0,1)))
                        self.teste, = self.XPDax.plot(x, y)
                        break
                    except:
                        pass
        except:
            pass
        ##          
        self.include_toolkit()

        self.include_LatticeParams().setGeometry(*self.LatticeParams_geo())

        self.include_E().setGeometry(*self.E_geo())

        self.include_CrystalSize().setGeometry(*self.Size_geo())
        
        self.include_InfoFrame().setGeometry(*self.Info_geo())

        self.include_CrystalFigure().setGeometry(*self.Crystal_geo())

        self.include_BaseAtoms().setGeometry(*self.Atoms_geo())

        self.setWindowTitle("PXRD - Qt version 1.0.0")
        
        self.update(ul=True, xpd = True, en = True)
        
        
        
        self.rescale()
        self.update_arrow()

    def initial_parameters(self):
        #  Lattice parameters group
        self.LatticeParams = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        self.par0 =     {'a': 5.640,  'b':5.640,  'c':5.640,  'alpha':90.0,  'beta':90.0,  'gamma':90.0}
        self.par_min =  {'a': 3.00,  'b':3.00,  'c':3.00,  'alpha':40,  'beta':40,  'gamma':40}
        self.par_max =  {'a': 13.00, 'b':13.00, 'c':13.00, 'alpha':150, 'beta':150, 'gamma':150}
        self.par_name = {'a': 'a',   'b':'b',   'c':'c',   'alpha':'\u03b1', 'beta':'\u03b2', 'gamma':'\u03b3'}
        self.par_button_clicks = {'a':self.button_a, 'b':self.button_b, 'c':self.button_c, 'alpha':self.button_alpha, 'beta':self.button_beta, 'gamma':self.button_gamma}
        self.par_slider_change = {'a':self.slider_a, 'b':self.slider_b, 'c':self.slider_c, 'alpha':self.slider_alpha, 'beta':self.slider_beta, 'gamma':self.slider_gamma}

        # i max
        #self.Imax = 5.5e4

        # initial energy
        self.init_E = 8.
        self.E = self.init_E
        xu.energy(self.init_E*1000)

        # initial size
        self.init_D = 500
        self.D = self.init_D

        # initial atom
        self.initial_atom = 'Fe'
        self.last_atom = self.initial_atom

        # geometry proposed
        self.ini_w = 1280
        self.ini_h = 720
        self.window_w = self.ini_w
        self.window_h = self.ini_h
        self.crystal_h = 300
        self.figure_w = 801
        self.figure_h = 497
        self.space = 5
        self.space_ini = 3

        self.additional_atoms = 0
        self.Atom_types = {0:self.initial_atom}
        self.AddAtoms_label_At = {}
        self.AddAtoms_label_x = {}
        self.AddAtoms_label_y = {}
        self.AddAtoms_label_z = {}
        self.AddAtoms_label_pos_x = {}
        self.AddAtoms_label_pos_y = {}
        self.AddAtoms_label_pos_z = {}
        self.AddAtoms_slider_x = {}
        self.AddAtoms_slider_y = {}
        self.AddAtoms_slider_z = {}
        self.AddAtoms_entry_At = {}
        self.AddAtoms_color = {}
        self.AddAtoms_entry_pos_x = {}
        self.AddAtoms_entry_pos_y = {}
        self.AddAtoms_entry_pos_z = {}
        self.AddAtoms_groupBox = {}
        self.AddAtoms_groupBox_layout = {}

        # some random colors
        self.red_light = '#ffdddd'
        self.red = '#ff8888'
        self.red_red = '#ff0000'
        self.orange = '#ff8800'
        self.orange_light = '#ffdd00'
        self.yellow = '#dddd00'
        self.yellow_light = '#ffff00'
        self.yellow_dark = '#aaaa00'
        self.yellowgreen = '#88ff00'
        self.yellowgreen_light = '#ddff00'
        self.green_light = '#ddffdd'
        self.green = '#88ff88'
        self.green_dark = '#66cc66'
        self.greencyan = '#00ff88'
        self.greencyan_light = '#00ffdd'
        self.cyan = '#00bbbb'
        self.cyan_light = '#00eeee'
        self.cyan_dark = '#009999'
        self.cyanblue = '#0088ff'
        self.cyanblue_light = '#00ddff'
        self.blue = '#8888ff'
        self.blue_light = '#ddddff'
        self.blue_dark = '#6666cc'
        self.bluemagenta = '#8800ff'
        self.bluemagenta_light = '#dd00ff'
        self.magenta = '#aa00aa'
        self.magenta_light = '#ee99ee'
        self.magentared = '#ff0088'
        self.magentared_light = '#ff00dd'
        self.gray = '#aaaaaa'
        self.gray_light = '#dddddd'
        self.gb_bg_ini = '#dddddd'
        self.gb_bg_fin = '#ffffff'
        self.sl_col_0 = '#666666'
        self.sl_col_1 = '#999999'
        self.sl_col_2 = '#bbbbbb'
        self.sl_col_3 = '#dddddd'
        self.sl_stp_0 = '#aaaafa'
        self.sl_stp_1 = '#6666f6'
        self.crysface_color = self.cyan_light
        self.crysedge_color = self.red_red

        self.color_atom0 = self.blue
        self.colors = ['#000000', self.blue, self.green, self.orange, self.greencyan, self.bluemagenta, self.yellow, self.cyan, self.magenta]
        self.colors_0 = [self.gray, self.blue_dark, self.green_dark, self.orange, '#999999', self.bluemagenta, self.yellow, self.cyan, self.magenta]

        # unit cell definitions
        self.degree = np.pi/180.
        self.plotlimits = 11

        self.Fhkl = {}
        # from calculate_xpd function
        self.theta_i = 5.
        self.theta_f = 60.
        self.Qi = 4*np.pi*self.E/12.398*np.sin(self.theta_i*np.pi/360.)
        self.Qf = 4*np.pi*self.E/12.398*np.sin(self.theta_f*np.pi/360.)

        step = 0.01
        self.theta_range = np.arange (self.theta_i,self.theta_f,step)
        self.intensity = np.zeros (len(self.theta_range))
        self.list_of_hkl = []
        hkl_max = 6
        for h in range (-hkl_max,hkl_max):
            for k in range (-hkl_max,hkl_max):
                for l in range (-hkl_max,hkl_max):
                    if not ( h == 0 and k == 0 and l == 0): self.list_of_hkl.append([h,k,l])

        self.old_HKL_H = 1
        self.old_HKL_K = 0
        self.old_HKL_L = 0

    # geometry proposed and calculations based on resize event
    def resizeEvent(self, event):
        #print("Window has been resized")
        self.calculate_geometry()
        QMainWindow.resizeEvent(self, event)
    def XPD_Figure_geo(self):
        x = self.space
        y = self.space_ini
        w = int(self.figure_w/float(self.ini_w)*self.window_w)
        h = int(float(self.figure_h)/self.ini_h*self.window_h)
        return [x,y,w,h]
    def LatticeParams_geo(self):
        x = self.space
        y = int(float(self.figure_h)/self.ini_h*self.window_h)+self.space+self.space_ini
        w = int((int(self.figure_w/float(self.ini_w)*self.window_w)-self.space)/2)
        h = self.window_h-int(float(self.figure_h)/self.ini_h*self.window_h)-self.space_ini-2*self.space
        return [x,y,w,h]
    def E_geo(self):
        x = int((int(self.figure_w/float(self.ini_w)*self.window_w)-self.space)/2) + 2*self.space
        y = int(float(self.figure_h)/self.ini_h*self.window_h)+self.space+self.space_ini
        w = int((int(self.figure_w/float(self.ini_w)*self.window_w)-self.space)/2)
        h = int((self.window_h-int(float(self.figure_h)/self.ini_h*self.window_h)-self.space_ini-2*self.space-3*self.space)*2.5/6)
        return [x,y,w,h]
    def Size_geo(self):
        x = int((int(self.figure_w/float(self.ini_w)*self.window_w)-self.space)/2) + 2*self.space
        y = int(float(self.figure_h)/self.ini_h*self.window_h)+self.space+self.space_ini + int((self.window_h-int(float(self.figure_h)/self.ini_h*self.window_h)-self.space_ini-2*self.space-3*self.space)*2.5/6) + self.space
        w = int((int(self.figure_w/float(self.ini_w)*self.window_w)-self.space)/2)
        h = int((self.window_h-int(float(self.figure_h)/self.ini_h*self.window_h)-self.space_ini-2*self.space-3*self.space)*1.5/6)
        return [x,y,w,h]
    def Info_geo(self):
        x = int((int(self.figure_w/float(self.ini_w)*self.window_w)-self.space)/2) + 2*self.space
        a = int(float(self.figure_h)/self.ini_h*self.window_h)+self.space+self.space_ini + int((self.window_h-int(float(self.figure_h)/self.ini_h*self.window_h)-self.space_ini-2*self.space-3*self.space)*2.5/6) + self.space
        b = int((self.window_h-int(float(self.figure_h)/self.ini_h*self.window_h)-self.space_ini-2*self.space-3*self.space)*1.5/6)+ self.space
        y = a + b
        w = int((int(self.figure_w/float(self.ini_w)*self.window_w)-self.space)/2)
        h = int((self.window_h-int(float(self.figure_h)/self.ini_h*self.window_h)-self.space_ini-2*self.space-3*self.space)*2/6)
        return [x,y,w,h]
    def Crystal_geo(self):
        x = int(self.figure_w/float(self.ini_w)*self.window_w)+2*self.space
        y = self.space_ini
        w = self.window_w-x-self.space
        h = int(float(self.crystal_h)/self.ini_h*self.window_h)
        return [x,y,w,h]
    def Atoms_geo(self):
        x = int(self.figure_w/float(self.ini_w)*self.window_w)+2*self.space
        y = int(float(self.crystal_h)/self.ini_h*self.window_h) + self.space_ini+self.space
        w = self.window_w-x-self.space
        h = self.window_h-int(float(self.crystal_h)/self.ini_h*self.window_h)-2*self.space-self.space_ini
        return [x,y,w,h]
    def calculate_geometry(self):
        self.window_w = self.width()
        self.window_h = self.height()
        self.space = 5
        self.space_ini = 3
        self.XPDFigure_groupBox.setGeometry(*self.XPD_Figure_geo())
        self.LatticeParams_groupBox.setGeometry(*self.LatticeParams_geo())
        self.E_groupBox.setGeometry(*self.E_geo())
        self.CrystalSize_groupBox.setGeometry(*self.Size_geo())
        self.CrystalFigure_groupBox.setGeometry(*self.Crystal_geo())
        self.BaseAtoms_groupBox.setGeometry(*self.Atoms_geo())
        self.Info_groupBox.setGeometry(*self.Info_geo())

    def eventFilter(self, source, event, flag = 0):
        if flag == 0:
            if event.type() == QEvent.Enter and source is self.XPDFigure_groupBox: self.Info_label.setText(names[version]['xpd'])
        elif flag in self.LatticeParams:
            if event.type() == QEvent.Enter and source is self.LatticeParams_button[flag]: self.Info_label.setText(names[version]['lattpar'])
            elif event.type() == QEvent.Enter and source is self.LatticeParams_slider[flag]: self.Info_label.setText(names[version]['lattpar'])
            elif event.type() == QEvent.Enter and source is self.LatticeParams_entry[flag]: self.Info_label.setText(names[version]['lattpar'])
        #elif 
        
        
        
        #print (event.type(), 
        if event.type() == QEvent.Leave: self.Info_label.setText('')
        
        
        #LatticeParams_button[self. LatticeParams[i]], i, 21)
        #     LatticeParams_layout.addWidget(self. LatticeParams_slider[self. LatticeParams[i]], i, 22, 1, 6)
        #     LatticeParams_layout.addWidget(self. LatticeParams_entry
        
        
        return super(Window, self).eventFilter(source, event)

    def include_XPDFigure(self):
        # just the name of the group
        self.XPDFigure_groupBox = QGroupBox(names[version]['title'], self)
        self.XPDFigure_groupBox.setFont(QFont(font,fontsize))
        group_opts = self.GroupBox_StyleSheet(self.green, self.green_light)
        self.XPDFigure_groupBox.setStyleSheet(group_opts)
        self.XPDFigure_groupBox.installEventFilter(self)
        
        # matplotlib figure and ax.
        self.XPDFigure, self.XPDax = plt.subplots()
        plt.subplots_adjust(left=0.08,right=0.98, bottom=0.08, top=0.98)
        
        self.main_plot, = self.XPDax.plot (self.theta_range,self.intensity, 'ko-',lw = 2.5, markersize = 4)
        self.colored_plots = []
        for i in ['r','g','b']:
            a, = self.XPDax.plot (self.theta_range,self.intensity, '{}o-'.format(i),lw = 1.2, markersize = 2)
            a.set_visible(False)
            self.colored_plots.append(a)
            
        #self.XPDax.set_ylim(-5,self.Imax)
        
        self.hkl_text = self.XPDax.text(5, 5, '')
        self.line, = self.XPDax.plot([0,0],[0,0], 'r--', lw = 1)
        
        # Canvas Widget that displays the `figure`; it takes the `figure` instance as a parameter to its __init__
        self.XPDcanvas = FigureCanvas(self.XPDFigure) 
        
        # Navigation widget; it takes the Canvas widget and a parent
        self.tool_manager = ToolManager(self.XPDFigure)
        self.XPDtoolbar = ToolbarQt(self.tool_manager, self.XPDFigure_groupBox) 
        

        
        # Setting the layout:
        Figure_layout = QVBoxLayout()
        Figure_layout.addSpacing(5)
        Figure_layout.addWidget(self.XPDtoolbar)
        Figure_layout.addWidget(self.XPDcanvas)
        self.XPDFigure_groupBox.setLayout(Figure_layout)
        
        return self.XPDFigure_groupBox
    def include_toolkit(self):
        backend_tools.add_tools_to_manager(self.tool_manager)
        backend_tools.add_tools_to_container(self.XPDtoolbar)
        for i in range(3):
            self.tool_manager.add_tool('Freeze_{}'.format(i), Freeze, curve=i, plot = self.colored_plots[i], main_plot = self.main_plot)
            self.XPDtoolbar.add_tool('Freeze_{}'.format(i),'foo')
        self.tool_manager.add_tool('Rescale', Rescale_y, func = self.rescale, main_plot = self.main_plot)
        self.XPDtoolbar.add_tool('Rescale','foo')
        
        self.include_HKL()
    def include_HKL(self):
        self.text_H = QLabel("  H ")
        self.text_K = QLabel("  K ")
        self.text_L = QLabel("  L ")
        self.text_H.setStyleSheet('QLabel {font: bold 12px}')
        self.text_K.setStyleSheet('QLabel {font: bold 12px}')
        self.text_L.setStyleSheet('QLabel {font: bold 12px}')

        self.le_H = QLineEdit(str(self.old_HKL_H))
        self.le_K = QLineEdit(str(self.old_HKL_K))
        self.le_L = QLineEdit(str(self.old_HKL_L))
        self.le_H.setMaxLength(2)
        self.le_K.setMaxLength(2)
        self.le_L.setMaxLength(2)
        self.le_H.setAlignment(Qt.AlignCenter)
        self.le_K.setAlignment(Qt.AlignCenter)
        self.le_L.setAlignment(Qt.AlignCenter)
        entry_opts = self.LEtool_StyleSheet(self.magenta_light)
        self.le_H.setStyleSheet(entry_opts)
        self.le_K.setStyleSheet(entry_opts)
        self.le_L.setStyleSheet(entry_opts)
        self.le_H.editingFinished.connect(self.update_HKL_H)
        self.le_K.editingFinished.connect(self.update_HKL_K)
        self.le_L.editingFinished.connect(self.update_HKL_L)
        self.le_H.setToolTip(names[version]['hkl_inf'] + names[version]['hkl_inf2'])
        self.le_K.setToolTip(names[version]['hkl_inf'] + names[version]['hkl_inf2'])
        self.le_L.setToolTip(names[version]['hkl_inf'] + names[version]['hkl_inf2'])

        self.showHKL_check = QCheckBox('')
        check_opts = self.Checkbox_StyleSheet(color = self.magenta_light)
        self.showHKL_check.setStyleSheet(check_opts)
        self.showHKL_check.setChecked(False)
        self.showHKL_check.stateChanged.connect(self.check_showHKL_TF)
        self.showHKL_check.setToolTip(names[version]['showHKL'])

        self.XPDtoolbar.addWidget(self.showHKL_check)
        self.XPDtoolbar.addWidget(self.text_H)
        self.XPDtoolbar.addWidget(self.le_H)
        self.XPDtoolbar.addWidget(self.text_K)
        self.XPDtoolbar.addWidget(self.le_K)
        self.XPDtoolbar.addWidget(self.text_L)
        self.XPDtoolbar.addWidget(self.le_L)

    def update_HKL_H(self):
        val = self.le_H.text()
        try:
            val_ = int(val)
            if val_ <-9 or val_ > 9:

                text = '"{}" {} {}'.format(val_, names[version]['problem_a'], 'H')
                self.Update_Info_label(text = text, bkg = self.red)
                self.le_H.setText(str(self.old_HKL_H))
            elif val_ == 0 and int(self.le_K.text()) == 0 and int(self.le_L.text()) == 0:
                text = '{}'.format(names[version]['hkl_000'])
                self.Update_Info_label(text = text, bkg = self.red)
                self.le_H.setText(str(self.old_HKL_H))

            else:
                self.old_HKL_H = val_
                self.update(ul=False, xpd = True, en = False)
        except:
            text = '"{}" {} {}'.format(val, names[version]['problem_a'], 'H')
            self.Update_Info_label(text = text, bkg = self.red)
            self.le_H.setText(str(self.old_HKL_H))
    def update_HKL_K(self):
        val = self.le_K.text()
        try:
            val_ = int(val)
            if val_ <-9 or val_ > 9:
                text = '"{}" {} {}'.format(val_, names[version]['problem_a'], 'K')
                self.Update_Info_label(text = text, bkg = self.red)
                self.le_K.setText(str(self.old_HKL_K))
            elif val_ == 0 and int(self.le_H.text()) == 0 and int(self.le_L.text()) == 0:
                text = '{}'.format(names[version]['hkl_000'])
                self.Update_Info_label(text = text, bkg = self.red)
                self.le_K.setText(str(self.old_HKL_K))
            else:
                self.old_HKL_K = val_
                self.update(ul=False, xpd = True, en = False)
        except:
            text = '"{}" {} {}'.format(val, names[version]['problem_a'], 'K')
            self.Update_Info_label(text = text, bkg = self.red)
            self.le_K.setText(str(self.old_HKL_K))
    def update_HKL_L(self):
        val = self.le_L.text()
        try:
            val_ = int(val)
            if val_ <-9 or val_ > 9:
                text = '"{}" {} {}'.format(val_, names[version]['problem_a'], 'L')
                self.Update_Info_label(text = text, bkg = self.red)
                self.le_L.setText(str(self.old_HKL_L))
            elif val_ == 0 and int(self.le_H.text()) == 0 and int(self.le_K.text()) == 0:
                text = '{}'.format(names[version]['hkl_000'])
                self.Update_Info_label(text = text, bkg = self.red)
                self.le_L.setText(str(self.old_HKL_L))
            else:
                self.old_HKL_L = val_
                self.update(ul=False, xpd = True, en = False)
        except:
            text = '"{}" {} {}'.format(val, names[version]['problem_a'],'L')
            self.Update_Info_label(text = text, bkg = self.red)
            self.le_L.setText(str(self.old_HKL_L))
    def update_arrow(self):
        h = self.old_HKL_H
        k = self.old_HKL_K
        l = self.old_HKL_L
        Q = self.Qhkl(h, k, l)
        _2theta = self.Q2tth (Q)
        ymax = self.XPDax.get_ylim()[1]
        try: 
            self.arrow_.remove()
        except:
            pass
        arrow = Arrow(_2theta,ymax*0.93,0,-ymax*0.09, color="#aa0088")
        if self.showHKL_check.isChecked():
            self.arrow_ = self.XPDax.add_patch(arrow)
            self.line.set_data([_2theta, _2theta], [0, ymax*0.77])
            self.hkl_text.set_text('{} {} {}'.format(h, k, l))
            self.hkl_text.set_position((_2theta, ymax*0.95))
        else:
            self.line.set_data([0,0],[0,0])
            self.hkl_text.set_text('')
    def check_showHKL_TF(self):
        self.update(ul=False, xpd = True, en = False)
        if self.showHKL_check.isChecked(): self.calc_HKL_planes()

    def include_LatticeParams(self):
        # just the name of the group
        self.LatticeParams_groupBox = QGroupBox(names[version]['pars'], self)
        self.LatticeParams_groupBox.setFont(QFont(font,fontsize))
        group_opts = self.GroupBox_StyleSheet(self.green, self.green_light)
        self.LatticeParams_groupBox.setStyleSheet(group_opts)
        #self.LatticeParams_groupBox.installEventFilter(self)
        
        
        #include buttons for restart, sliders and labels 
        self. LatticeParams_button = {}
        button_opts = self.PushButton_StyleSheet(self.green)

        self. LatticeParams_slider = {}
        slider_opts = self.Slider_StyleSheet()
        
        self. LatticeParams_entry = {}
        entry_opts = self.LineEdit_StyleSheet(self.green)

        for i in self. LatticeParams:
            self.LatticeParams_button.update({i:QPushButton(self.par_name[i])})
            self.LatticeParams_button[i].setFont(QFont(font,fontsize))
            self.LatticeParams_button[i].clicked.connect(self.par_button_clicks[i])
            self.LatticeParams_button[i].setStyleSheet(button_opts)
            self.LatticeParams_button[i].setToolTip(self.par_name[i] + ': ' + names[version]['tt_b_pars'])
            if i in ['a', 'b', 'c']:
                self. LatticeParams_slider.update({i:DoubleSlider(3, Qt.Horizontal)})
            else:
                self. LatticeParams_slider.update({i:DoubleSlider(1, Qt.Horizontal)})
            self.LatticeParams_slider[i].setStyleSheet(slider_opts)
            self. LatticeParams_slider[i].setMinimum(self.par_min[i])
            self. LatticeParams_slider[i].setMaximum(self.par_max[i])
            self. LatticeParams_slider[i].setValue(self.par0[i])
            self. LatticeParams_slider[i].setSingleStep(1)
            self. LatticeParams_slider[i].setTickPosition(QSlider.NoTicks)
            self. LatticeParams_slider[i].valueChanged.connect(self.par_slider_change[i])
            self. LatticeParams_slider[i].setToolTip(names[version]['tt_s_pars'])

            self.LatticeParams_entry.update({i:QLineEdit(str(self.par0[i]))})
            self.LatticeParams_entry[i].setStyleSheet(entry_opts)
            self.LatticeParams_entry[i].setFont(QFont(font,fontsize))
            self.LatticeParams_entry[i].setMaxLength(5)
            self.LatticeParams_entry[i].editingFinished.connect(lambda par = i: self.update_sliders(par))
            self.LatticeParams_entry[i].setFixedWidth(60)
            self.LatticeParams_entry[i].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.LatticeParams_entry[i].setToolTip(names[version]['tt_e_pars'])
            
            #self.LatticeParams_button[i].installEventFilter(self, i)
            #self.LatticeParams_slider[i].installEventFilter(self, i)
            #self.LatticeParams_entry[i].installEventFilter(self, i)


        LatticeParams_layout = QGridLayout()
        for i in range(6):
             #LatticeParams_layout.addSpacing(5)
             LatticeParams_layout.addWidget(self. LatticeParams_button[self. LatticeParams[i]], i, 21)
             LatticeParams_layout.addWidget(self. LatticeParams_slider[self. LatticeParams[i]], i, 22, 1, 6)
             LatticeParams_layout.addWidget(self. LatticeParams_entry[self. LatticeParams[i]], i, 29)
        self.LatticeParams_groupBox.setLayout( LatticeParams_layout)
        
        return self.LatticeParams_groupBox       
    def button_a (self): 
        self. LatticeParams_slider['a'].setValue(self.par0['a'])
        self. LatticeParams_entry['a'].setText(str(self. LatticeParams_slider['a'].value()))
        self.update(ul=True, en = False)
    def button_b (self): 
        self. LatticeParams_slider['b'].setValue(self.par0['b'])
        self. LatticeParams_entry['b'].setText(str(self. LatticeParams_slider['b'].value()))
        self.update(ul=True, en = False)
    def button_c (self): 
        self. LatticeParams_slider['c'].setValue(self.par0['c'])
        self. LatticeParams_entry['c'].setText(str(self. LatticeParams_slider['c'].value()))
        self.update(ul=True, en = False)
    def button_alpha (self): 
        self. LatticeParams_slider['alpha'].setValue(self.par0['alpha'])
        self. LatticeParams_entry['alpha'].setText(str(self. LatticeParams_slider['alpha'].value()))
        self.update(ul=True, en = False)
    def button_beta  (self): 
        self. LatticeParams_slider['beta'].setValue(self.par0['beta'])
        self. LatticeParams_entry['beta'].setText(str(self. LatticeParams_slider['beta'].value()))
        self.update(ul=True, en = False)
    def button_gamma (self): 
        self. LatticeParams_slider['gamma'].setValue(self.par0['gamma'])
        self. LatticeParams_entry['gamma'].setText(str(self. LatticeParams_slider['gamma'].value()))
        self.update(ul=True, en = False)
    def slider_a (self): 
        self.LatticeParams_entry['a'].setText(str(self.LatticeParams_slider['a'].value()))
        self.update(ul=True, en = False)
    def slider_b (self): 
        self.LatticeParams_entry['b'].setText(str(self. LatticeParams_slider['b'].value()))
        self.update(ul=True, en = False)
    def slider_c (self): 
        self.LatticeParams_entry['c'].setText(str(self. LatticeParams_slider['c'].value()))
        self.update(ul=True, en = False)
    def slider_alpha (self): 
        self.LatticeParams_entry['alpha'].setText(str(self. LatticeParams_slider['alpha'].value()))
        self.update(ul=True, en = False)
    def slider_beta  (self): 
        self.LatticeParams_entry['beta'].setText(str(self. LatticeParams_slider['beta'].value()))
        self.update(ul=True, en = False)
    def slider_gamma (self): 
        self.LatticeParams_entry['gamma'].setText(str(self. LatticeParams_slider['gamma'].value()))
        self.update(ul=True, en = False)
    def update_sliders(self, par):
        val = self.LatticeParams_entry[par].text()
        try:
            self.LatticeParams_slider[par].setValue(float(val))
            self.update(ul=True, en = False)
        except:
            text = '"{}" {} {}'.format(val, names[version]['problem_a'], names[version][par])
            self.Update_Info_label(text = text, bkg = self.red)
            self.LatticeParams_entry[par].setText(str(self.LatticeParams_slider[par].value()))

    def include_CrystalFigure(self):
        # just the name of the group
        self.CrystalFigure_groupBox = QGroupBox(names[version]['crystal'], self) 
        self.CrystalFigure_groupBox.setFont(QFont(font,fontsize))
        group_opts = self.GroupBox_StyleSheet(self.gray, self.gray_light)
        self.CrystalFigure_groupBox.setStyleSheet(group_opts)

        self.CrystalFigure = plt.figure()
        self.Crystalax = plt.axes([0.08, 0.08, 0.9, 0.9],projection='3d')
        self.Crystalcanvas = FigureCanvas(self.CrystalFigure) 

        self.Vis_slider = DoubleSlider(1,Qt.Vertical)
        slider_opts = self.Slider_StyleSheet('v')
        self.Vis_slider.setStyleSheet(slider_opts)
        self.Vis_slider.setMinimum(5)
        self.Vis_slider.setMaximum(13)
        self.Vis_slider.setValue(10)
        self.Vis_slider.setTickPosition(QSlider.NoTicks)
        self.Vis_slider.valueChanged.connect(self.change_limits)
        self.Vis_slider.setToolTip(names[version]['tt_s_uc'])

        self.Edge_check = QCheckBox('')
        #self.Edge_check.setFont(QFont(font,fontsize, 5))
        check_opts = self.Checkbox_StyleSheet(color = '#ff0000')
        self.Edge_check.setStyleSheet(check_opts)
        self.Edge_check.setChecked(True)
        self.Edge_check.stateChanged.connect(self.check_edge_TF)
        self.Edge_check.setToolTip(names[version]['crysedge'])
        
        self.Face_check = QCheckBox('')
        check_opts = self.Checkbox_StyleSheet(color = '#00ffff')
        self.Face_check.setStyleSheet(check_opts)
        self.Face_check.setChecked(True)
        self.Face_check.stateChanged.connect(self.check_face_TF)
        self.Face_check.setToolTip(names[version]['crysface'])

        self.Atoms_check = QCheckBox('')
        check_opts = self.Checkbox_StyleSheet(color = self.blue_dark)
        self.Atoms_check.setStyleSheet(check_opts)
        self.Atoms_check.setChecked(True)
        self.Atoms_check.stateChanged.connect(self.check_showhideatoms_TF)
        self.Atoms_check.setToolTip(names[version]['crysatoms'])

        # Setting the layout:
        Figure_layout = QGridLayout()
        Figure_layout.addWidget(self.Crystalcanvas, 0, 1, 6, 10)
        Figure_layout.addWidget(self.Vis_slider, 0, 0, 4, 1)
        Figure_layout.addWidget(self.Atoms_check, 4, 0, 1, 1)
        Figure_layout.addWidget(self.Edge_check, 5, 0, 1, 1)
        Figure_layout.addWidget(self.Face_check, 6, 0, 1, 1)
        self.CrystalFigure_groupBox.setLayout(Figure_layout)
        
        return self.CrystalFigure_groupBox
    def change_limits(self):
        self.plotlimits = self.Vis_slider.value()
        self.update(ul=False, xpd = False, en = False)
    def check_edge_TF(self):
        self.update(ul=False, xpd = False, en = False)
    def check_face_TF(self):
        self.update(ul=False, xpd = False, en = False)
    def check_showhideatoms_TF(self):
        self.update(ul=False, xpd = False, en = False)



    def calc_HKL_planes(self):
        h = int(self.le_H.text())
        k = int(self.le_K.text())
        l = int(self.le_L.text())
        lim = self.plotlimits
        lim2 = lim/5.
        mesh = np.arange(-0.2,1.21, 0.2)
        x0, x1 = np.meshgrid(mesh, mesh)
        d = []
        delta = 0.1
        diag = np.array(self.set_pos(1,1,1))+delta
        #dist = np.sqrt(np.dot(diag,[h,k,l]))
        #d_HKL = 2*np.pi/self.Qhkl(h,k,l)
        _alpha = 0.75
        #while n*d_HKL <= 5*dist: n+=1
        if h != 0:
            for i in range(-4,4,1): 
                x = (-x0*k -x1*l +i)/h
                [xx, yy, zz] = self.set_pos(x,x0,x1)
                if np.all([np.any(xx <= diag[0]), np.any(xx >= -delta), np.any(yy <= diag[1]), np.any(yy >= -delta), np.any(zz <= diag[2]), np.any(zz >= -delta)]):
                    self.curv = self.Crystalax.plot_surface(xx, yy, zz, color = self.magenta_light, alpha = _alpha )
        elif k != 0:
            for i in range(-4,4,1): 
                y = (-x0*h -x1*l +i)/k
                [xx, yy, zz] = self.set_pos(x0,y,x1)
                if np.all([np.any(xx <= diag[0]), np.any(xx >= -delta), np.any(yy <= diag[1]), np.any(yy >= -delta), np.any(zz <= diag[2]), np.any(zz >= -delta)]):
                    self.curv = self.Crystalax.plot_surface(xx, yy, zz, color = self.magenta_light, alpha = _alpha )
        elif l != 0:
            for i in range(-4,4,1): 
                z = (-x0*h -x1*k +i)/l
                [xx, yy, zz] = self.set_pos(x0,x1,z)
                if np.all([np.any(xx <= diag[0]), np.any(xx >= -delta), np.any(yy <= diag[1]), np.any(yy >= -delta), np.any(zz <= diag[2]), np.any(zz >= -delta)]):
                    self.curv = self.Crystalax.plot_surface(xx, yy, zz, color = self.magenta_light, alpha = _alpha )
        self.Crystalcanvas.draw()

    def include_E(self):
        # just the name of the group
        self.E_groupBox = QGroupBox(names[version]['wvl'], self)
        self.E_groupBox.setFont(QFont(font,fontsize))
        group_opts = self.GroupBox_StyleSheet(self.red, self.red_light)
        self.E_groupBox.setStyleSheet(group_opts)
        
        self.E_button = QPushButton('E (keV)')
        self.E_button.clicked.connect(self.reset_E)
        button_opts = self.PushButton_StyleSheet(self.red)
        self.E_button.setStyleSheet(button_opts)
        self.E_button.setToolTip(names[version]['en_0'] + ': ' + names[version]['tt_b_pars'])
        
        self.Wvl_button = QPushButton('\u03bb (\u212b)')
        self.Wvl_button.clicked.connect(self.reset_E)
        self.Wvl_button.setStyleSheet(button_opts)
        self.Wvl_button.setToolTip(names[version]['wvl_0'] + ': ' + names[version]['tt_b_pars'])
        
        self.E_slider = DoubleSlider(4,Qt.Horizontal)
        slider_opts = self.Slider_StyleSheet()
        self.E_slider.setStyleSheet(slider_opts)
        self.E_slider.setMinimum(4)
        self.E_slider.setMaximum(20)
        self.E_slider.setValue(self.E)
        self.E_slider.setTickPosition(QSlider.NoTicks)
        self.E_slider.valueChanged.connect(self.E_slider_change)
        self.E_slider.setToolTip(names[version]['tt_s_pars'])

        self.Wvl_slider = DoubleSlider(4,Qt.Horizontal)
        self.Wvl_slider.setStyleSheet(slider_opts)
        self.Wvl_slider.setMinimum(12.398/20.)
        self.Wvl_slider.setMaximum(12.398/4.)
        self.Wvl_slider.setValue(12.398/self.E)
        self.Wvl_slider.setTickPosition(QSlider.NoTicks)
        self.Wvl_slider.valueChanged.connect(self.Wvl_slider_change)
        self.Wvl_slider.setToolTip(names[version]['tt_s_pars'])

        self.E_entry = QLineEdit()
        entry_opts = self.LineEdit_StyleSheet(self.red)

        self.E_entry.setFont(QFont(font,fontsize))
        self.E_entry.setStyleSheet(entry_opts)
        self.E_entry.setText('{: 5.2f}'.format(self.E))
        self.E_entry.setMaxLength(5)
        self.E_entry.setFixedWidth(60)
        self.E_entry.setToolTip(names[version]['tt_e_pars'])

        self.E_entry.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.E_entry.editingFinished.connect(self.E_update_slider)

        self.Wvl_entry = QLineEdit()
        self.Wvl_entry.setFont(QFont(font,fontsize))
        self.Wvl_entry.setStyleSheet(entry_opts)
        self.Wvl_entry.setText('{: 5.2f}'.format(12.398/self.E))
        self.Wvl_entry.setMaxLength(5)
        self.Wvl_entry.setFixedWidth(60)
        self.Wvl_entry.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.Wvl_entry.editingFinished.connect(self.Wvl_upadate_slider)
        self.Wvl_entry.setToolTip(names[version]['tt_e_pars'])

        E_layout = QGridLayout()
        E_layout.addWidget(self.E_button, 0, 0)
        E_layout.addWidget(self.E_slider, 0, 1, 1, 5)
        E_layout.addWidget(self.E_entry, 0, 6)
        E_layout.addWidget(self.Wvl_button, 1, 0)
        E_layout.addWidget(self.Wvl_slider, 1, 1, 1, 5)
        E_layout.addWidget(self.Wvl_entry, 1, 6)
        
        self.E_groupBox.setLayout(E_layout)
        
        return self.E_groupBox
    def reset_E(self):
        self.E = self.init_E
        self.E_slider.setValue(self.E)
        self.Wvl_slider.setValue(12.398/self.E)
        self.updateQs()
        self.update(ul=True, en = True)
    def E_slider_change(self):
        self.E = self.E_slider.value()
        self.E_entry.setText('{: 5.2f}'.format(self.E))
        self.Wvl_entry.setText('{: 5.2f}'.format(12.398/self.E))
        self.Wvl_slider.setValue(12.398/self.E)
        self.update(ul=True, en = True)
    def Wvl_slider_change(self):
        self.Wvl = self.Wvl_slider.value()
        self.Wvl_entry.setText('{: 5.2f}'.format(12.398/self.E))
        self.E_entry.setText('{: 5.2f}'.format(self.E))
        self.E_slider.setValue(12.398/self.Wvl)
        self.updateQs()
        self.update(ul=True, en = True)
    def E_update_slider(self):
        val = self.E_entry.text()
        try:
            self.E = float(val)
            self.E_slider.setValue(self.E)
            self.Wvl_slider.setValue(12.398/self.E)
            self.Wvl_entry.setText('{: 5.2f}'.format(12.398/self.E))
            self.updateQs()
            self.update(ul=True, en = True)
        except:
            text = '"{}" {} {}'.format(val, names[version]['problem_a'], names[version]['en_0'])
            self.Update_Info_label(text = text, bkg = self.red)
            self.E_entry.setText('{: 5.2f}'.format(self.E))
    def Wvl_upadate_slider(self):
        val = self.Wvl_entry.text()
        try:
            self.E = 12.398/float(val)
            self.E_slider.setValue(self.E)
            self.Wvl_slider.setValue(float(self.Wvl_entry.text()))
            self.E_entry.setText('{: 5.2f}'.format(self.E))
            self.updateQs()
            self.update(ul=True, en = True)
        except:
            text = '"{}" {} {}'.format(val, names[version]['problem_a'], names[version]['wvl_0'])
            self.Update_Info_label(text = text, bkg = self.red)
            self.Wvl_entry.setText('{: 5.2f}'.format(12.398/self.E))
    def updateQs(self):
        self.Qi = self.Q(self.theta_i)
        self.Qf = self.Q(self.theta_f)
    def Q(self,_2theta): return 4*np.pi/self.Wvl_slider.value()*np.sin(_2theta*np.pi/360.)

    def include_CrystalSize(self):
        # just the name of the group
        self.CrystalSize_groupBox = QGroupBox(names[version]['size'], self) 
        self.CrystalSize_groupBox.setFont(QFont(font,fontsize))
        group_opts = self.GroupBox_StyleSheet(self.red, self.red_light)
        self.CrystalSize_groupBox.setStyleSheet(group_opts)
        
        self.CrystalSize_button = QPushButton('D (\u212b)')
        self.CrystalSize_button.clicked.connect(self.reset_D)
        button_opts = self.PushButton_StyleSheet(self.red)
        self.CrystalSize_button.setStyleSheet(button_opts)
        self.CrystalSize_button.setToolTip(names[version]['size_0'] + ': ' + names[version]['tt_b_pars'])

        self.CrystalSize_slider = DoubleSlider(1,Qt.Horizontal)
        slider_opts = self.Slider_StyleSheet()
        self.CrystalSize_slider.setStyleSheet(slider_opts)
        self.CrystalSize_slider.setMinimum(15)
        self.CrystalSize_slider.setMaximum(1000)
        self.CrystalSize_slider.setValue(self.init_D)
        self.CrystalSize_slider.setTickPosition(QSlider.NoTicks)
        self.CrystalSize_slider.valueChanged.connect(self.D_slider_change)
        self.CrystalSize_slider.setToolTip(names[version]['tt_s_pars'])
        
        self.CrystalSize_entry = QLineEdit()
        entry_opts = self.LineEdit_StyleSheet(self.red)
        self.CrystalSize_entry.setFont(QFont(font,fontsize))
        self.CrystalSize_entry.setStyleSheet(entry_opts)
        self.CrystalSize_entry.setText('{: 6.1f}'.format(self.init_D))
        self.CrystalSize_entry.setMaxLength(6)
        self.CrystalSize_entry.setFixedWidth(60)
        self.CrystalSize_entry.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.CrystalSize_entry.editingFinished.connect(self.Size_update_slider)
        self.CrystalSize_entry.setToolTip(names[version]['tt_e_pars'])
        
        CrystalSize_layout = QHBoxLayout()
        CrystalSize_layout.addSpacing(5)
        CrystalSize_layout.addWidget(self.CrystalSize_button)
        CrystalSize_layout.addWidget(self.CrystalSize_slider)
        CrystalSize_layout.addWidget(self.CrystalSize_entry)

        self.CrystalSize_groupBox.setLayout(CrystalSize_layout)

        return self.CrystalSize_groupBox
    def reset_D(self):
        self.D = self.init_D
        self.CrystalSize_slider.setValue(self.D)
    def D_slider_change(self):
        self.D = self.CrystalSize_slider.value()
        self.CrystalSize_entry.setText('{: 6.1f}'.format(self.D))
        self.update(ul=False, xpd = True, en = False)
    def Size_update_slider(self):
        val = self.CrystalSize_entry.text()
        try:
            self.D = float(val)
            self.CrystalSize_slider.setValue(float(self.CrystalSize_entry.text()))
            self.update(ul=False, xpd = True, en = False)
        except:
            text = '"{}" {} {}'.format(val, names[version]['problem_a'], names[version]['size_0'])
            self.Update_Info_label(text = text, bkg = self.red)
            self.CrystalSize_entry.setText('{: 6.1f}'.format(self.D))

    def include_InfoFrame(self):
        self.Info_groupBox = QGroupBox(names[version]['info'], self) 
        self.Info_groupBox.setFont(QFont(font,fontsize))
        group_opts = self.GroupBox_StyleSheet(self.red, self.red_light)
        self.Info_groupBox.setStyleSheet(group_opts)
        
        self.Info_label = QLabel()
        self.Info_label.setFont(QFont(font,fontsize))
        self.Info_label.setText(names[version]['welcome'])
        self.Info_label.setMinimumWidth(150)
        label_opts = 'QLabel { background: #d4d4d4; border-radius: 3px;}'
        self.Info_label.setStyleSheet(label_opts)
        self.Info_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        
        Info_layout = QVBoxLayout()
        Info_layout.addSpacing(5)
        Info_layout.addWidget(self.Info_label)
        
        self.Info_groupBox.setLayout(Info_layout)
        
        return self.Info_groupBox
    def Update_Info_label(self, text = '', bkg = '#d4d4d4'):
        label_opts = r'QLabel { background: ' + '{}'.format(bkg) + r'; border-radius: 3px;}'
        self.Info_label.setStyleSheet(label_opts)
        self.Info_label.setText(text)

    def include_BaseAtoms(self):
        # just the name of the group
        self.BaseAtoms_groupBox = QGroupBox(names[version]['base'], self)
        self.BaseAtoms_groupBox.setFont(QFont(font,fontsize))
        self.BaseAtoms_groupBox.setStyleSheet(self.GroupBox_StyleSheet(self.blue, self.blue_light))

        self.BaseAtoms_entry_0 = QLineEdit(self.initial_atom)
        self.BaseAtoms_entry_0.setStyleSheet(self.LineEdit_StyleSheet(self.blue))
        self.BaseAtoms_entry_0.setFont(QFont(font,fontsize))
        self.BaseAtoms_entry_0.setMaxLength(2)
        self.BaseAtoms_entry_0.editingFinished.connect(lambda atom = 0: self.newAtomEntered(atom))
        self.BaseAtoms_entry_0.setFixedWidth(40);
        self.BaseAtoms_entry_0.setToolTip(names[version]['label_0'])
        
        self.BaseAtoms_color = QPushButton('')
        self.BaseAtoms_color.clicked.connect(self.getBaseAtom_color)
        button_opts = self.PushButton_StyleSheet(self.blue)
        button_opts2 = self.PushButton_StyleSheet2(self.blue)
        self.BaseAtoms_color.setStyleSheet(button_opts2)
        self.BaseAtoms_color.setToolTip(names[version]['color_0'])

        self.BaseAtoms_button_addAtom = QPushButton('+ {}'.format(names[version]['atm']))
        self.BaseAtoms_button_addAtom.setFont(QFont(font,fontsize))
        self.BaseAtoms_button_addAtom.clicked.connect(self.add_Atom)
        self.BaseAtoms_button_addAtom.setStyleSheet(button_opts)
        self.BaseAtoms_button_addAtom.setToolTip(names[version]['addAtoms'])
        self.BaseAtoms_button_remAtom = QPushButton('- {}'.format(names[version]['atm']))
        self.BaseAtoms_button_remAtom.setFont(QFont(font,fontsize))
        self.BaseAtoms_button_remAtom.clicked.connect(self.rem_Atom)
        self.BaseAtoms_button_remAtom.setStyleSheet(button_opts)
        self.BaseAtoms_button_remAtom.setToolTip(names[version]['remAtoms'])

        self.BaseAtoms_tabs = QTabWidget()



        # North  = 0; South  = 1; West  = 2; East = 3
        pos_ = 2
        if pos_ == 0:
            f = 'QTabBar::tab:!selected {margin-left: 2px; }'
            c3 = 'min-width: 15ex;padding: 2px;}'
            b = 'QTabWidget::tab-bar { left: 15px;}'
            c2 = 'border-bottom-color: #C2C7CB; border-top-left-radius: 5px; border-top-right-radius: 5px;'
            
        if pos_ == 2:
            f = 'QTabBar::tab:!selected {margin-top: 2px; }'
            c3 = 'min-height: 15ex;padding: 2px;}'
            b = 'QTabWidget::tab-bar { left: 0px; top: 15px}'
            c2 = 'border-bottom-color: #C2C7CB; border-top-left-radius: 5px; border-bottom-left-radius: 5px;'
        
        self.BaseAtoms_tabs.setTabPosition(pos_)
        
        a = 'QTabWidget::pane { border-top: 2px solid #C2C7CB; border-left: 2px solid #C2C7CB; border-right: 2px solid #C2C7CB; border-bottom: 2px solid #C2C7CB;}'
        c = 'QTabBar::tab { background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #E1E1E1, '
        c1 = 'stop: 0.4 #DDDDDD, stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3); border: 2px solid #C4C4C3; '
        
        d = 'QTabBar::tab:selected, QTabBar::tab:hover { background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,'
        d1 = 'stop: 0 {}, stop:  1.0 {});'.format(self.blue,self.blue_light)+'}'
        e = 'QTabBar::tab:selected {border-color: '+ '{};border-bottom-color: {}'.format(self.blue, self.blue)+';}'
        tab_opts = a + b + c + c1+ c2 + c3 + d + d1 + e + f
        self.BaseAtoms_tabs.setStyleSheet(tab_opts)

        
        self.BaseAtoms_tab1 = QWidget()
        self.BaseAtoms_tab2 = QWidget()
        self.BaseAtoms_tab3 = QWidget()

        self.layout = [QVBoxLayout(self.BaseAtoms_tab1),QVBoxLayout(self.BaseAtoms_tab2),QVBoxLayout(self.BaseAtoms_tab3)]
        
        self.BaseAtoms_tabs.addTab(self.BaseAtoms_tab1,"pos 1-3")
        self.BaseAtoms_tabs.addTab(self.BaseAtoms_tab2,"pos 4-6")
        self.BaseAtoms_tabs.addTab(self.BaseAtoms_tab3,"pos 7-9")
        for i in range (3): self.BaseAtoms_tabs.setTabEnabled (i, False)
        
        BaseAtoms_layout = QGridLayout()
        BaseAtoms_layout.setContentsMargins(5, 15, 5, 5)
        BaseAtoms_layout.addWidget(self.BaseAtoms_entry_0, 1, 0)
        BaseAtoms_layout.addWidget(self.BaseAtoms_color,1,2)
        BaseAtoms_layout.addWidget(self.BaseAtoms_button_addAtom, 1, 10,1,2)
        BaseAtoms_layout.addWidget(self.BaseAtoms_button_remAtom, 1, 12,1,2)
        BaseAtoms_layout.addWidget(self.BaseAtoms_tabs,2,0,-1,-1)
        self.BaseAtoms_groupBox.setLayout(BaseAtoms_layout)

        return self.BaseAtoms_groupBox      
    def newAtomEntered(self, atom):
        #a = 'QLineEdit {border: 2px solid #' 
        #b = '; border-radius: 10px; padding: 0 8px; background: #8888ff; selection-background-color: 6666ee;} '
        if atom == 0:
            func = self.BaseAtoms_entry_0
        else:
            func = self.AddAtoms_entry_At[atom]
        try:
            eval('xu.materials.elements.{}'.format(func.text()))
            self.Atom_types.update({atom:func.text()})
            self.last_atom = func.text()
            self.update(ul=False, xpd = True, en = True)
        except:
            text0 = func.text()
            func.setText(self.last_atom)
            text = '"{}" {} {}'.format(text0, names[version]['problem_a'], names[version]['atm'])
            self.Update_Info_label(text = text, bkg = self.red)
    def add_Atom(self):
        self.additional_atoms += 1
        if self.additional_atoms>9: 
            print ('maximum of 10 atoms')
            self.additional_atoms = 9
        else: 
            self.include_atom()
            self.rescale()
    def rem_Atom(self):
        self.additional_atoms -= 1
        if self.additional_atoms < 0: self.additional_atoms = 0
        else: 
            self.exclude_atom()
            self.rescale()
    def include_atom(self):
        i = self.additional_atoms
        j = (i-1)//3
        k = (i-1)%3

        self.BaseAtoms_tabs.setTabEnabled (j, True)
        self.BaseAtoms_tabs.setCurrentIndex(j)
        

        self.AddAtoms_groupBox.update({i:QGroupBox('{} {}'.format(names[version]['atm'],i))})
        self.AddAtoms_groupBox[i].setFont(QFont(font,fontsize))
        self.AddAtoms_groupBox[i].setStyleSheet(self.GroupBox_StyleSheet(self.colors_0[i-1],'#dddddd'))
        self.AddAtoms_groupBox[i].setMaximumHeight(int(self.layout[j].geometry().getRect()[3]/3))
        self.layout[j].addWidget(self.AddAtoms_groupBox[i])

        self.AddAtoms_groupBox_layout.update({i:QGridLayout(self.AddAtoms_groupBox[i])})

        self.AddAtoms_label_x.update({i:QLabel()})
        self.AddAtoms_label_x[i].setFont(QFont(font,fontsize))
        self.AddAtoms_label_x[i].setText('{}'.format('x'))
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_label_x[i],0,1)

        self.AddAtoms_label_y.update({i:QLabel()})
        self.AddAtoms_label_y[i].setFont(QFont(font,fontsize))
        self.AddAtoms_label_y[i].setText('{}'.format('y'))
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_label_y[i],1,1)
        self.AddAtoms_label_z.update({i:QLabel()})
        self.AddAtoms_label_z[i].setFont(QFont(font,fontsize))
        self.AddAtoms_label_z[i].setText('{}'.format('z'))
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_label_z[i],2,1)
        
        self.AddAtoms_slider_x.update({i:DoubleSlider(3,Qt.Horizontal)})
        slider_opts = self.Slider_StyleSheet()
        self.AddAtoms_slider_x[i].setStyleSheet(slider_opts)
        self.AddAtoms_slider_x[i].setMinimum(0)
        self.AddAtoms_slider_x[i].setMaximum(1)
        self.AddAtoms_slider_x[i].setValue(0.5)
        self.AddAtoms_slider_x[i].setTickPosition(QSlider.NoTicks)
        self.AddAtoms_slider_x[i].valueChanged.connect(lambda value, atom = i: self.pos_x_slider_change(value, atom))
        self.AddAtoms_slider_x[i].setToolTip(names[version]['atm'] + '{}, '.format(i) + names[version]['posx'] + ': '.format(i) + names[version]['tt_s_pars'])
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_slider_x[i],0,2,1,7)
        
        self.AddAtoms_entry_pos_x.update({i:QLineEdit('0.5')})
        entry_opts = self.LineEdit_StyleSheet(self.colors_0[i-1])
        self.AddAtoms_entry_pos_x[i].setStyleSheet(entry_opts)
        self.AddAtoms_entry_pos_x[i].setFont(QFont(font,fontsize))
        self.AddAtoms_entry_pos_x[i].setMaxLength(5)
        self.AddAtoms_entry_pos_x[i].editingFinished.connect(lambda atom = i: self.new_pos_x(atom))
        self.AddAtoms_entry_pos_x[i].setFixedWidth(60)
        self.AddAtoms_entry_pos_x[i].setToolTip(names[version]['atm'] + '{}, '.format(i) + names[version]['posx'] + ': '.format(i) + names[version]['tt_e_pars'])
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_entry_pos_x[i],0,11)

        self.AddAtoms_slider_y.update({i:DoubleSlider(3,Qt.Horizontal)})
        self.AddAtoms_slider_y[i].setStyleSheet(slider_opts)
        self.AddAtoms_slider_y[i].setMinimum(0)
        self.AddAtoms_slider_y[i].setMaximum(1)
        self.AddAtoms_slider_y[i].setValue(0.5)
        self.AddAtoms_slider_y[i].setTickPosition(QSlider.NoTicks)
        self.AddAtoms_slider_y[i].valueChanged.connect(lambda value, atom = i: self.pos_y_slider_change(value, atom))
        self.AddAtoms_slider_y[i].setToolTip(names[version]['atm'] + '{}, '.format(i) + names[version]['posy'] + ': '.format(i) + names[version]['tt_s_pars'])

        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_slider_y[i],1,2,1,7)

        self.AddAtoms_entry_pos_y.update({i:QLineEdit('0.5')})
        self.AddAtoms_entry_pos_y[i].setStyleSheet(entry_opts)
        self.AddAtoms_entry_pos_y[i].setFont(QFont(font,fontsize))
        self.AddAtoms_entry_pos_y[i].setMaxLength(5)
        self.AddAtoms_entry_pos_y[i].editingFinished.connect(lambda atom = i: self.new_pos_y(atom))
        self.AddAtoms_entry_pos_y[i].setFixedWidth(60)
        self.AddAtoms_entry_pos_y[i].setToolTip(names[version]['atm'] + '{}, '.format(i) + names[version]['posy'] + ': '.format(i) + names[version]['tt_e_pars'])
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_entry_pos_y[i],1,11)

        self.AddAtoms_slider_z.update({i:DoubleSlider(3,Qt.Horizontal)})
        self.AddAtoms_slider_z[i].setStyleSheet(slider_opts)
        self.AddAtoms_slider_z[i].setMinimum(0)
        self.AddAtoms_slider_z[i].setMaximum(1)
        self.AddAtoms_slider_z[i].setValue(0.5)
        self.AddAtoms_slider_z[i].setTickPosition(QSlider.NoTicks)
        self.AddAtoms_slider_z[i].valueChanged.connect(lambda value, atom = i: self.pos_z_slider_change(value, atom))
        self.AddAtoms_slider_z[i].setToolTip(names[version]['atm'] + '{}, '.format(i) + names[version]['posz'] + ': '.format(i) + names[version]['tt_s_pars'])
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_slider_z[i],2,2,1,7)

        self.AddAtoms_entry_pos_z.update({i:QLineEdit('0.5')})
        self.AddAtoms_entry_pos_z[i].setStyleSheet(entry_opts)
        self.AddAtoms_entry_pos_z[i].setFont(QFont(font,fontsize))
        self.AddAtoms_entry_pos_z[i].setMaxLength(5)
        self.AddAtoms_entry_pos_z[i].editingFinished.connect(lambda atom = i: self.new_pos_z(atom))
        self.AddAtoms_entry_pos_z[i].setFixedWidth(60)
        self.AddAtoms_entry_pos_z[i].setToolTip(names[version]['atm'] + '{}, '.format(i) + names[version]['posz'] + ': '.format(i) + names[version]['tt_e_pars'])
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_entry_pos_z[i],2,11)

        self.AddAtoms_entry_At.update({i:QLineEdit(self.last_atom)})
        self.AddAtoms_entry_At[i].setStyleSheet(entry_opts)
        self.AddAtoms_entry_At[i].setFont(QFont(font,fontsize))
        self.AddAtoms_entry_At[i].setMaxLength(2)
        self.AddAtoms_entry_At[i].editingFinished.connect(lambda atom = i: self.newAtomEntered(atom))
        self.AddAtoms_entry_At[i].setFixedWidth(40)
        self.AddAtoms_entry_At[i].setToolTip(names[version]['atomtype'])
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_entry_At[i],0,0)
        self.Atom_types.update({i:self.last_atom})
        
        self.AddAtoms_color.update({i:QPushButton('')})
        self.AddAtoms_color[i].clicked.connect(lambda tf, atom = i: self.getAddAtoms_color(tf, atom))
        button_opts = self.PushButton_StyleSheet2(self.colors_0[i-1])
        self.AddAtoms_color[i].setStyleSheet(button_opts)
        self.AddAtoms_color[i].setToolTip(names[version]['color_0'])
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_color[i],1,0)

        self.update()
    def pos_x_slider_change(self, value, atom):
        self.AddAtoms_entry_pos_x[atom].setText('{: 5.3f}'.format(self.AddAtoms_slider_x[atom].value()))
        self.update(ul=False, xpd = True, en = False)
    def pos_y_slider_change(self, value, atom):
        self.AddAtoms_entry_pos_y[atom].setText('{: 5.3f}'.format(self.AddAtoms_slider_y[atom].value()))
        self.update(ul=False, xpd = True, en = False)
    def pos_z_slider_change(self, value, atom):
        self.AddAtoms_entry_pos_z[atom].setText('{: 5.3f}'.format(self.AddAtoms_slider_z[atom].value()))
        self.update(ul=False, xpd = True, en = False)
    def new_pos_x(self,atom):
        val = self.AddAtoms_entry_pos_x[atom].text()
        try:
            self.AddAtoms_slider_x[atom].setValue(float(val))
            self.update(ul=False, xpd = True, en = False)
        except:
            text = '"{}" {} {}'.format(val, names[version]['problem_a'], names[version]['atm_0'])
            self.Update_Info_label(text = text, bkg = self.red)
            self.AddAtoms_entry_pos_x[atom].setText('{: 5.3f}'.format(self.AddAtoms_slider_x[atom].value()))
    def new_pos_y(self,atom):
        val = self.AddAtoms_entry_pos_y[atom].text()
        try:
            self.AddAtoms_slider_y[atom].setValue(float(val))
            self.update(ul=False, xpd = True, en = False)
        except:
            text = '"{}" {} {}'.format(val, names[version]['problem_a'], names[version]['atm_0'])
            self.Update_Info_label(text = text, bkg = self.red)
            self.AddAtoms_entry_pos_y[atom].setText('{: 5.3f}'.format(self.AddAtoms_slider_y[atom].value()))
    def new_pos_z(self,atom):
        val = self.AddAtoms_entry_pos_z[atom].text()
        try:
            self.AddAtoms_slider_z[atom].setValue(float(val))
            self.update(ul=False, xpd = True, en = False)
        except:
            text = '"{}" {} {}'.format(val, names[version]['problem_a'], names[version]['atm_0'])
            self.Update_Info_label(text = text, bkg = self.red)
            self.AddAtoms_entry_pos_z[atom].setText('{: 5.3f}'.format(self.AddAtoms_slider_z[atom].value()))
    def exclude_atom(self):
        i = self.additional_atoms
        j = (i)//3
        k = (i)%3
        self.layout[j].itemAt(k).widget().deleteLater()
        self.update(ul=False, xpd = True, en = False)

    def GroupBox_StyleSheet(self, color_name_ini, color_name_fin):
        a = 'QGroupBox {background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 ' + '{}, stop: 1 {});'.format(self.gb_bg_ini, self.gb_bg_fin)
        b = 'border: 2px solid gray; border-radius: 5px; margin-top: 2ex;}'
        c = 'QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top center; '
        d = 'padding: 0 3px; background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 {}, stop: 1 {});'.format(color_name_ini, color_name_fin)+'}'
        return a + b + c + d
    def Slider_StyleSheet(self, orientation = 'h'):
        if orientation == 'h':
            a = 'QSlider::groove {border: 1px solid '+'{}; height: 8px; background: qlineargradient(x1:0,'.format(self.sl_col_1)
            b = 'y1:0, x2:0, y2:1, stop:0 {}, stop:1 {});  margin: 2px 0;'.format(self.sl_col_2, self.sl_col_3)+'}'
            c = 'QSlider::handle {background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 '+'{}, stop:1 {});'.format(self.sl_stp_0, self.sl_stp_1)
            d = 'border: 1px solid {}'.format(self.sl_col_0)+ '; width: 12px; height: 16px; margin: -4px 0; border-radius: 3px;}'
            return a + b + c + d
        if orientation == 'v':
            a = 'QSlider::groove {border: 1px solid '+'{}; width: 8px; background: qlineargradient(x1:0,'.format(self.sl_col_1)
            b = 'y1:0, x2:0, y2:1, stop:0 {}, stop:1 {});  margin: 2px 0;'.format(self.sl_col_2, self.sl_col_3)+'}'
            c = 'QSlider::handle {background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 '+'{}, stop:1 {});'.format(self.sl_stp_0, self.sl_stp_1)
            d = 'border: 1px solid {}'.format(self.sl_col_0)+ '; height: 12px; width: 16px; margin: 0 -4px; border-radius: 3px;}'
            return a + b + c + d
    def LineEdit_StyleSheet(self,bg):
        a = 'QLineEdit {border: 2px solid black; border-radius: 10px; padding: 0 8px; background: '+'{};'.format(bg)
        b = 'selection-background-color: {};'.format(bg)+'}'
        return a+b
    def LEtool_StyleSheet(self, bg):
        a = 'QLineEdit {border: 2px solid black; border-radius: 4px; padding: 1px -2px; background: '+'{};'.format(bg)
        b = 'selection-background-color: {};'.format(bg)+' font: bold 12px; max-width: 2em;}'
        return a+b        
    def PushButton_StyleSheet(self, bg):
        a = 'QPushButton {background-color: '+'{}; border-style: outset; border-width: 2px;'.format(bg) 
        b = 'border-radius: 10px; border-color: black; font: bold 12px; min-width: 5em; padding: 1px}'
        return a+b
    def PushButton_StyleSheet2(self, bg):
        a = 'QPushButton {background-color: '+'{}; border-style: outset; border-width: 2px;'.format(bg) 
        b = 'border-radius: 5px; padding: 1px}'
        return a+b
    def Checkbox_StyleSheet(self, color):
        a = 'QCheckBox::indicator {width: 17px;height: 17px; background-color: #ffffff; border-color: '+'{}'.format(color) + '; border-radius: 3px;border: 2px solid;}'
        b = 'QCheckBox::indicator::unchecked {width: 17px;height: 17px; border-radius: 3px; background-color: ' + '{}'.format('#ffffff')+';}'
        c = 'QCheckBox::indicator:unchecked:hover {width: 17px;height: 17px; border-radius: 3px; background-color: #ffffff;}'
        d = 'QCheckBox::indicator:unchecked:pressed {width: 17px;height: 17px; border-radius: 3px; background-color: #dddddd;}'
        e = 'QCheckBox::indicator::checked {width: 17px;height: 17px; border-radius: 3px; background-color: ' + '{}'.format(color)+';}'
        f = 'QCheckBox::indicator:checked:hover {width: 17px;height: 17px; border-radius: 3px; background-color: ' + '{}'.format(color)+';}'
        g = 'QCheckBox::indicator:checked:pressed {width: 17px;height: 17px; border-radius: 3px; background-color: #dddddd;}'
        return a + b + c + d + e + f + g

    def getBaseAtom_color(self):
        color = QColorDialog.getColor().name()
        self.BaseAtoms_entry_0.setStyleSheet(self.LineEdit_StyleSheet(color))
        self.BaseAtoms_color.setStyleSheet(self.PushButton_StyleSheet2(color))
        self.color_atom0 = color
        self.update(xpd = False, en = False)
    def getAddAtoms_color(self, tf, atom):
        color = QColorDialog.getColor().name()
        self.AddAtoms_color[atom].setStyleSheet(self.PushButton_StyleSheet2(color))
        self.AddAtoms_entry_At[atom].setStyleSheet(self.LineEdit_StyleSheet(color))
        self.colors_0[atom-1] = color
        self.AddAtoms_groupBox[atom].setStyleSheet(self.GroupBox_StyleSheet(color,'#dddddd'))
        self.AddAtoms_entry_pos_x[atom].setStyleSheet(self.LineEdit_StyleSheet(color))
        self.AddAtoms_entry_pos_y[atom].setStyleSheet(self.LineEdit_StyleSheet(color))
        self.AddAtoms_entry_pos_z[atom].setStyleSheet(self.LineEdit_StyleSheet(color))
        self.update(xpd = False, en = False)

    def update(self, ul=False, xpd = True, en = True):
        x_add = []
        y_add = []
        z_add = []
        pos_add = []
        for i in range (self.additional_atoms):
            x_add.append(self.AddAtoms_slider_x[i+1].value())
            y_add.append(self.AddAtoms_slider_y[i+1].value())
            z_add.append(self.AddAtoms_slider_z[i+1].value())
            pos_add.append([x_add[i], y_add[i], z_add[i]])
        
        self.update_unit_cell(pos_add, self.plotlimits)
        self.update_arrow()
        if ul: self.update_limits()
        if xpd: self.calculate_xpd(en)
        if self.showHKL_check.isChecked(): self.calc_HKL_planes()
    def rescale(self):
        a = self.intensity.max()
        self.XPDax.set_ylim(-5, a*1.2)
        self.update_arrow()
        self.XPDcanvas.draw()
    def update_unit_cell(self, pos_add, plotlimits):


        #def set_pos(x, y, z): return [x*a_su+y*b_su*cg+z*c_su*cb,y*b_su*sg+z*c_su*abg,z*c_su*np.sqrt(sb*sb-abg*abg)]
        x = []
        y = []
        z = []
        pos = []
        for _ii in range(2):
            for _jj in range (2):
                for _kk in range (2):
                    pos.append(self.set_pos(_ii, _jj, _kk))
        self.scatter = []
        colors = []
        for i in range (8):
            x.append(pos[i][0])
            y.append(pos[i][1])
            z.append(pos[i][2])
            colors.append(self.color_atom0)
        
        verts = [[pos[0],pos[1],pos[3],pos[2]], [pos[4],pos[5],pos[7],pos[6]], 
                [pos[0],pos[1],pos[5],pos[4]],  [pos[2],pos[3],pos[7],pos[6]], 
                [pos[1],pos[3],pos[7],pos[5]],  [pos[4],pos[6],pos[2],pos[0]]]

        self.Crystalax.cla()
        if self.Edge_check.isChecked(): lw_ = 2
        else: lw_ = 0
        if self.Face_check.isChecked(): alpha_ = 0.55
        else: alpha_ = 0
        self.verts = self.Crystalax.add_collection3d(Poly3DCollection(verts, facecolors = self.crysface_color, linewidths=lw_, edgecolors=self.crysedge_color, alpha=alpha_))
        self.Crystalax.set_xlabel('a')
        self.Crystalax.set_ylabel('b')
        self.Crystalax.set_zlabel('c')
        for i in range(len(pos_add)):
            res = self.set_pos(pos_add[i][0], pos_add[i][1], pos_add[i][2])
            x.append(res[0])
            y.append(res[1])
            z.append(res[2])
            colors.append(self.colors_0[i])
        if self.Atoms_check.isChecked(): self.scatter = self.Crystalax.scatter3D (x, y, z, color=colors, s = 300)
        #self.Crystalax.pbaspect = [1.0, 1.0, 1.5]
        self.Crystalax.set_box_aspect((1.0, 1.0, 1.0))

        self.Crystalax.set_xlim([-1,plotlimits])
        self.Crystalax.set_ylim([-1,plotlimits])
        self.Crystalax.set_zlim([-1,plotlimits])
        self.Crystalcanvas.draw()
    def set_pos(self,x,y,z): 
        a_su = self.LatticeParams_slider['a'].value()
        b_su = self.LatticeParams_slider['b'].value()
        c_su = self.LatticeParams_slider['c'].value()
        alpha_su = self.LatticeParams_slider['alpha'].value()
        beta_su = self.LatticeParams_slider['beta'].value()
        gamma_su = self.LatticeParams_slider['gamma'].value()
        sa = np.sin(alpha_su*self.degree)
        ca = np.cos(alpha_su*self.degree)
        sb = np.sin(beta_su*self.degree)
        cb = np.cos(beta_su*self.degree)
        sg = np.sin(gamma_su*self.degree)
        cg = np.cos(gamma_su*self.degree)
        abg = (ca-cg*cb)/sg
        return [x*a_su+y*b_su*cg+z*c_su*cb,y*b_su*sg+z*c_su*abg,z*c_su*np.sqrt(sb*sb-abg*abg)]

    def calculate_xpd(self, en):
        self.intensity = np.zeros (len(self.theta_range))
        for _ii in self.list_of_hkl_used:
            self.intensity += self.calculate_F2(_ii, en)*self.calculate_intensity(_ii)/(self.Qhkl(*_ii)*self.Qhkl(*_ii))
        self.main_plot.set_ydata(self.intensity)
        self.XPDcanvas.draw_idle()
    def update_limits(self):
        self.list_of_hkl_used = []
        for _ii in self.list_of_hkl:
            Q = self.Qhkl(*_ii)
            if (Q > self.Qi) and (Q < self.Qf):
                self.list_of_hkl_used.append(_ii)
    def Qhkl(self,h,k,l):
        a = self.LatticeParams_slider['a'].value()
        b = self.LatticeParams_slider['b'].value()
        c = self.LatticeParams_slider['c'].value()
        alpha = self.LatticeParams_slider['alpha'].value()
        beta = self.LatticeParams_slider['beta'].value()
        gamma = self.LatticeParams_slider['gamma'].value()
        ha = h/a
        kb = k/b
        lc = l/c
        sa = np.sin(alpha*self.degree)
        ca = np.cos(alpha*self.degree)
        sb = np.sin(beta*self.degree)
        cb = np.cos(beta*self.degree)
        sg = np.sin(gamma*self.degree)
        cg = np.cos(gamma*self.degree)
        return 2*np.pi*np.sqrt((ha*sa*ha*sa+ kb*sb*kb*sb+ lc*sg*lc*sg + 2*ha*kb*(ca*cb-cg)+ 2*ha*lc*(ca*cg-cb) + 2*kb*lc*(cb*cg-ca))/(1.-ca*ca-cb*cb-cg*cg+2*ca*cb*cg))
    def calculate_F2(self,hkl,en_):
        en = self.E*1000
        Q = self.Qhkl(*hkl)
        if False:
            atom = eval('xu.materials.elements.{}'.format(str(self.Atom_types[0])))
            Fhkl = atom.f0(Q)+ atom.f1(en)+1j*atom.f2(en)
            for i in range (self.additional_atoms):
                 atom = eval('xu.materials.elements.{}'.format(str(self.Atom_types[i+1])))
                 f = atom.f0(Q)+atom.f1(en)+1j*atom.f2(en)
                 pos = [self.AddAtoms_slider_x[i+1].value(),self.AddAtoms_slider_y[i+1].value(),self.AddAtoms_slider_z[i+1].value()]
                 Fhkl += f*np.exp(-2*np.pi*1j*np.dot(hkl,pos))
            return np.power(np.abs(Fhkl),2)
        else:
            s = 'h{}k{}l{}'.format(*hkl)
            if en_:
                atom = eval('xu.materials.elements.{}'.format(str(self.Atom_types[0])))
                Fhkl = atom.f0(Q)+ atom.f1(en)+1j*atom.f2(en)
                self.Fhkl.update({s:Fhkl})
                self.f = {}
            else:
                try:
                    Fhkl = self.Fhkl[s]
                except:
                    atom = eval('xu.materials.elements.{}'.format(str(self.Atom_types[0])))
                    Fhkl = atom.f0(Q)+ atom.f1(en)+1j*atom.f2(en)
                    self.Fhkl.update({s:Fhkl})
            for i in range (self.additional_atoms):
                 s2 = 'h{}k{}l{}i{}'.format(*hkl,i)
                 if en_:
                    atom = eval('xu.materials.elements.{}'.format(str(self.Atom_types[i+1])))
                    f = atom.f0(Q)+atom.f1(en)+1j*atom.f2(en)
                    self.f.update({s2:f})
                 else:
                    try:
                        f = self.f[s2]
                    except:
                        atom = eval('xu.materials.elements.{}'.format(str(self.Atom_types[i+1])))
                        f = atom.f0(Q)+atom.f1(en)+1j*atom.f2(en)
                        self.f.update({s2:f})
                 pos = [self.AddAtoms_slider_x[i+1].value(),self.AddAtoms_slider_y[i+1].value(),self.AddAtoms_slider_z[i+1].value()]
                 Fhkl += f*np.exp(-2*np.pi*1j*np.dot(hkl,pos))
        return np.power(np.abs(Fhkl),2)
    def calculate_intensity(self,hkl): 
        tth = self.Q2tth(self.Qhkl(*hkl))
        w = self.size_width(tth)*360/np.pi
        return 1./(np.sqrt(2*np.pi)*w)*np.exp(-np.power(self.theta_range-tth,2)/(2.*w*w))
    def Q2tth(self,Q): return 360./np.pi*np.arcsin(Q*self.Wvl_slider.value()/(4*np.pi))
    def size_width(self,tth): return 0.9*self.Wvl_slider.value()/(2.355*self.CrystalSize_slider.value()*np.cos(tth/2.*self.degree))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    a = Icons()
    clock = Window()
    clock.show()
    sys.exit(app.exec_())

