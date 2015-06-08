'''
Created on 02/06/2015

@author: andre
'''

from specmorph.model import create_model_images, BDModel
from specmorph.util import logger
from specmorph.geometry import ellipse_params

import wx
from wx.lib.newevent import NewEvent
import matplotlib
from pycasso.util import radialProfile
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx as Toolbar
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
import numpy as np
from copy import copy, deepcopy

__all__ = ['InitialModelFrame']


################################################################################
ParamUpdateEvent, EVT_PARAM_UPDATE_EVENT = NewEvent()
ModelUpdateEvent, EVT_MODEL_UPDATE_EVENT = NewEvent()
################################################################################


################################################################################
class PlotPanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        
        self.plotSetup()
        self.figure = Figure()
        self.figureCanvas = FigureCanvas(self, wx.ID_ANY, self.figure)
        
        toolbar = Toolbar(self.figureCanvas)
        toolbar.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.figureCanvas, 1, wx.EXPAND)
        sizer.Add(toolbar, 0, wx.LEFT)
        self.SetSizer(sizer)


    def plotSetup(self):
        plotpars = {'legend.fontsize': 8,
                    'xtick.labelsize': 10,
                    'ytick.labelsize': 10,
                    'text.fontsize': 10,
                    'axes.titlesize': 12,
                    'lines.linewidth': 0.5,
                    'font.family': 'sans serif',
                    'image.cmap': 'GnBu',
                }
        matplotlib.rcParams.update(plotpars)

    def createImageAxes(self, data, subplot, nticks=5, vmin=None, vmax=None):
        axes = self.figure.add_subplot(subplot)
        image = axes.imshow(data, vmin=vmin, vmax=vmax)
        cbar = self.figure.colorbar(image, ax=axes)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nticks))
        axes.set_xticks([])
        axes.set_yticks([])
        return axes, image

    
    def reset(self, title, obs_im, mod_im, res_im, r, obs_r, mod_r, bulge_r, disk_r):
        self.figure.clear()
        self.figure.suptitle(title)
        gs = GridSpec(2, 3, height_ratios=[2.0, 3.0])

        vmin = np.nanmin(obs_im)
        vmax = np.nanmax(obs_im)

        self.obsAxes, self.obsImage = self.createImageAxes(obs_im, gs[0,0], vmin=vmin, vmax=vmax)
        self.modAxes, self.modImage = self.createImageAxes(mod_im, gs[0,1], vmin=vmin, vmax=vmax)
        self.resAxes, self.resImage = self.createImageAxes(res_im, gs[0,2])
        
        self.obsAxes.set_title('observed')
        self.modAxes.set_title('model')
        self.resAxes.set_title('residual')

        vmin = obs_r.min()
        vmax = obs_r.max()
        pad = (vmax - vmin) * 0.1
        vmin -= pad
        vmax += pad

        self.radprofAxes = self.figure.add_subplot(gs[1,:])
        self.radprofAxes.set_ylim(vmin, vmax)
        self.radprofAxes.set_xlim(r.min(), r.max())

        self.radprofAxes.set_ylabel('log flux')
        self.radprofAxes.set_xlabel('radius [pixel]')
        
        self.obsLines = self.radprofAxes.plot(r, obs_r, 'k-', label='observed')
        self.modLines = self.radprofAxes.plot(r, mod_r, 'k:', label='model')
        self.bulgeLines = self.radprofAxes.plot(r, bulge_r, 'r:', label='bulge')
        self.diskLines = self.radprofAxes.plot(r, disk_r, 'b:', label='disk')
        self.radprofAxes.legend(loc='upper right')
        self.figureCanvas.draw()

        
    def update(self, obs_im, mod_im, res_im, r, obs_r, mod_r, bulge_r, disk_r):
        self.obsImage.set_data(obs_im)
        self.modImage.set_data(mod_im)
        self.resImage.set_data(res_im)

        self.obsLines.pop(0).remove()
        self.modLines.pop(0).remove()
        self.bulgeLines.pop(0).remove()
        self.diskLines.pop(0).remove()
        self.obsLines = self.radprofAxes.plot(r, obs_r, 'k-')
        self.modLines = self.radprofAxes.plot(r, mod_r, 'k:')
        self.bulgeLines = self.radprofAxes.plot(r, bulge_r, 'r:')
        self.diskLines = self.radprofAxes.plot(r, disk_r, 'b:')

        self.figureCanvas.draw()
################################################################################


################################################################################
class ParamCtrl(wx.Panel):
    maxSlider = 2000
    
    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args, **kwargs)

        self.name = 'noname'
        self.value = 0.0
        self.llimit = 0.0
        self.ulimit = 1.0

        self.label = wx.StaticText(self, wx.ID_ANY, self.name)
        self.valueText = wx.TextCtrl(self, wx.ID_ANY, '')
        self.llimitText = wx.TextCtrl(self, wx.ID_ANY, '')
        self.ulimitText = wx.TextCtrl(self, wx.ID_ANY, '')
        self.slider = wx.Slider(self, wx.ID_ANY, 0, 0, self.maxSlider)

        self.slider.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.onSlide)
        self.slider.Bind(wx.EVT_COMMAND_SCROLL_CHANGED, self.onSlide)
        self.valueText.Bind(wx.EVT_TEXT, self.onValueChange)
        self.llimitText.Bind(wx.EVT_TEXT, self.onLimitsChange)
        self.ulimitText.Bind(wx.EVT_TEXT, self.onLimitsChange)

        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer1.Add(self.label, 0, wx.CENTER, 5)
        sizer1.Add(self.valueText, 0, wx.LEFT, 5)
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2.Add(self.llimitText, 0, wx.LEFT, 5)
        sizer2.Add(self.slider, 1, wx.EXPAND, 5)
        sizer2.Add(self.ulimitText, 0, wx.LEFT, 5)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(sizer1, 1, wx.ALL, 5)
        sizer.Add(sizer2, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(sizer)
        
    
    def init(self, param):
        self.name = param.name        
        self.value = param.value        
        self.llimit = param.limits[0]
        self.ulimit = param.limits[1]
        self.label.SetLabel(self.name)
        self.valueText.ChangeValue(str(self.value))
        self.llimitText.ChangeValue(str(self.llimit))
        self.ulimitText.ChangeValue(str(self.ulimit))
        self.setSlider(self.value)
        

    def setSlider(self, val):
        pos = self.value2position(val)
        self.slider.SetValue(pos)
        
        
    def setValue(self, val):
        self.value = val
        self.valueText.ChangeValue(str(val))
        self.setSlider(val)

    
    def sendChangeEvent(self):
        event = ParamUpdateEvent(param=self.name, value=self.value,
                                 llimit=self.llimit, ulimit=self.ulimit)
        wx.PostEvent(self.GetEventHandler(), event)
        

    def onValueChange(self, event):
        val_str = event.GetString()
        try:
            val = float(val_str)
        except:
            self.valueText.SetBackgroundColour('RED')
            return
        self.valueText.SetBackgroundColour(None)
        self.value = val
        self.setSlider(val)
        self.sendChangeEvent()


    def onLimitsChange(self, event):
        try:
            self.llimit = float(self.llimitText.GetValue())
        except:
            self.llimitText.SetBackgroundColour('RED')
            return
        try:
            self.ulimit = float(self.ulimitText.GetValue())
        except:
            self.ulimitText.SetBackgroundColour('RED')
            return
        self.llimitText.SetBackgroundColour(None)
        self.ulimitText.SetBackgroundColour(None)
        self.setSlider(self.value)


    def onSlide(self, event):
        pos = event.GetPosition()
        self.value = self.position2value(pos)
        
        self.valueText.ChangeValue(str(self.value))
        self.llimitText.ChangeValue(str(self.llimit))
        self.ulimitText.ChangeValue(str(self.ulimit))
        self.valueText.SetBackgroundColour(None)
        self.llimitText.SetBackgroundColour(None)
        self.ulimitText.SetBackgroundColour(None)
        self.sendChangeEvent()
        

    def position2value(self, pos):
        return self.llimit + pos * (self.ulimit - self.llimit) / self.maxSlider


    def value2position(self, val):
        return int(self.maxSlider * (val - self.llimit) / (self.ulimit - self.llimit))
################################################################################


################################################################################
class ControlPanel(wx.Panel):
    def __init__(self, parent, wxid, model, model_file, *args, **kwargs):
        wx.Panel.__init__(self, parent, wxid, *args, **kwargs)
        
        self.originalModel = model
        self.modelFile = model_file
        self.model = deepcopy(self.originalModel)

        self.resetButton = wx.Button(self, wx.ID_ANY, 'Reset', (-1, -1), wx.DefaultSize)
        self.saveButton = wx.Button(self, wx.ID_ANY, 'Save', (-1, -1), wx.DefaultSize)

        self.params = {}
        self.X0 = self.addParamCtrl(self.model.x0)
        self.I_e = self.addParamCtrl(self.model.bulge.I_e)
        self.r_e = self.addParamCtrl(self.model.bulge.r_e)
        self.n = self.addParamCtrl(self.model.bulge.n)

        self.Y0 = self.addParamCtrl(self.model.y0)
        self.I_0 = self.addParamCtrl(self.model.disk.I_0)
        self.h = self.addParamCtrl(self.model.disk.h)
        
        self.resetButton.Bind(wx.EVT_BUTTON, self.onResetButton)
        self.saveButton.Bind(wx.EVT_BUTTON, self.onSaveButton)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer_l = wx.BoxSizer(wx.VERTICAL)
        sizer_l.Add(self.X0, 1, wx.EXPAND | wx.ALL, 5)
        sizer_l.Add(self.I_e, 1, wx.EXPAND | wx.ALL, 5)
        sizer_l.Add(self.r_e, 1, wx.EXPAND | wx.ALL, 5)
        sizer_l.Add(self.n, 1, wx.EXPAND | wx.ALL, 5)

        sizer_r = wx.BoxSizer(wx.VERTICAL)
        sizer_r.Add(self.Y0, 1, wx.EXPAND | wx.ALL, 5)
        sizer_r.Add(self.I_0, 1, wx.EXPAND | wx.ALL, 5)
        sizer_r.Add(self.h, 1, wx.EXPAND | wx.ALL, 5)
        sizer_r.Add(self.saveButton, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        sizer_r.Add(self.resetButton, 0, wx.ALIGN_CENTER | wx.ALL, 5)

        sizer.Add(sizer_l, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(sizer_r, 1, wx.EXPAND | wx.ALL, 5)

        self.SetSizer(sizer)
        

    def addParamCtrl(self, param):
        paramCtrl = ParamCtrl(self, wx.ID_ANY)
        self.initParamCtrl(paramCtrl, param)
        paramCtrl.Bind(EVT_PARAM_UPDATE_EVENT, self.onParamUpdate)
        return paramCtrl


    def initParamCtrl(self, paramCtrl, param):
        paramCtrl.init(param)
        self.params[param.name] = param


    def sendModelUpdateEvent(self):
        event = ModelUpdateEvent(model=self.model)
        wx.PostEvent(self.GetEventHandler(), event)
    

    def onSaveButton(self, event):
        self.originalModel = deepcopy(self.model)
        with open(self.modelFile, 'w') as f:
            logger.debug('Saving model file %s.' % self.modelFile)
            try:
                f.write(str(self.originalModel))
            except:
                logger.warn('Could not write cache model file %s' % self.modelFile)


    def onResetButton(self, event):
        self.model = deepcopy(self.originalModel)
        self.params = {}
        
        self.initParamCtrl(self.X0, self.model.x0)
        self.initParamCtrl(self.Y0, self.model.y0)

        self.initParamCtrl(self.I_e, self.model.bulge.I_e)
        self.initParamCtrl(self.r_e, self.model.bulge.r_e)
        self.initParamCtrl(self.n, self.model.bulge.n)

        self.initParamCtrl(self.I_0, self.model.disk.I_0)
        self.initParamCtrl(self.h, self.model.disk.h)
        self.sendModelUpdateEvent()

        
    def onParamUpdate(self, event):
        param = self.params[event.param]
        param.setValue(event.value)
        param.setLimits(event.llimit, event.ulimit)
        self.sendModelUpdateEvent()
################################################################################


################################################################################
class InitialModelFrame(wx.Frame):
    def __init__(self, parent, wxid, flux, noise, psf, model_file, title, plot_title, *args, **kwargs):
        wx.Frame.__init__(self, parent, wxid, title, *args, **kwargs)
        
        self.flux = flux
        self.noise = noise
        self.psf = psf
        
        self.plotTitle = plot_title
        
        self.modelFile = model_file
        self.loadModel()
        
        self.plotPanel = PlotPanel(self)
        self.controlPanel = ControlPanel(self, wx.ID_ANY, self.model, self.modelFile)

        self.controlPanel.Bind(EVT_MODEL_UPDATE_EVENT, self.onModelUpdate)

        frameSizer = wx.BoxSizer(wx.VERTICAL)
        frameSizer.Add(self.plotPanel, 1, wx.EXPAND)
        frameSizer.Add(self.controlPanel, 0, wx.EXPAND)
        self.SetSizerAndFit(frameSizer)

        self.SetAutoLayout(True)
        
        self.updatePlots(self.initialModel, reset=True)    
    
    
    def updatePlots(self, model, reset=False):
        bulge_flux, disk_flux = create_model_images(model, self.flux.shape, self.psf)
        model_flux = bulge_flux + disk_flux
        
        x0 = model.x0.value - 1.0
        y0 = model.y0.value - 1.0
        pa, ell = ellipse_params(self.flux, x0, y0)
        pa = (90.0 + pa) * np.pi / 180.0
        ba = 1.0 - ell
        bins = np.arange(30)
        r = bins[:-1] + 0.5 
        
        obs_r = radialProfile(self.flux, bins, x0, y0, pa, ba, rad_scale=1.0)
        obs_r = np.log10(obs_r)

        mod_r = radialProfile(model_flux, bins, x0, y0, pa, ba, rad_scale=1.0)
        mod_r = np.log10(mod_r)

        bulge_r = radialProfile(bulge_flux, bins, x0, y0, pa, ba, rad_scale=1.0)
        bulge_r = np.log10(bulge_r)

        disk_r = radialProfile(disk_flux, bins, x0, y0, pa, ba, rad_scale=1.0)
        disk_r = np.log10(disk_r)

        obs_im = np.log10(self.flux)
        mod_im = np.log10(model_flux)
        res_im = obs_im - mod_im

        if reset:
            self.plotPanel.reset(self.plotTitle, obs_im, mod_im, res_im, r, obs_r, mod_r, bulge_r, disk_r)
        else:
            self.plotPanel.update(obs_im, mod_im, res_im, r, obs_r, mod_r, bulge_r, disk_r)
    
        
    def setModelFile(self, model_file):
        self.modelFile = model_file
        self.loadModel()
    
    
    def loadModel(self):
        self.initialModel = BDModel.load(self.modelFile)
        self.model = deepcopy(self.initialModel)
        
        
    def onModelUpdate(self, event):
        self.updatePlots(event.model)
################################################################################
    
