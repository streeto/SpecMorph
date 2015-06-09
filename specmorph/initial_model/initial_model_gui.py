'''
Created on 02/06/2015

@author: andre
'''

from specmorph.model import create_model_images, BDModel
from specmorph.util import logger
from specmorph.geometry import ellipse_params, distance, r50
from specmorph.fitting import fit_image

import wx
from wx.lib.newevent import NewEvent
import matplotlib
from pycasso.util import radialProfile
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx as Toolbar
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np
from copy import deepcopy
from os import path

__all__ = ['InitialModelFrame']


################################################################################
ParamUpdateEvent, EVT_PARAM_UPDATE_EVENT = NewEvent()
ModelUpdateEvent, EVT_MODEL_UPDATE_EVENT = NewEvent()
FitEvent, EVT_FIT_EVENT = NewEvent()
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

    def createImageAxes(self, data, subplot, nticks=5, vmin=None, vmax=None, cmap=None):
        axes = self.figure.add_subplot(subplot)
        image = axes.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
        self.figure.colorbar(image, ax=axes)
        axes.set_xticks([])
        axes.set_yticks([])
        return axes, image

    
    def reset(self, title, obs_im, mod_im, res_im, r, obs_r, mod_r, bulge_r, disk_r):
        self.figure.clear()
        self.figure.suptitle(title)
        gs = GridSpec(2, 3, height_ratios=[2.0, 3.0])

        vmin = np.nanmin(obs_im)
        vmax = np.nanmax(obs_im)
        res_amp = np.abs(res_im).max()

        self.obsAxes, self.obsImage = self.createImageAxes(obs_im, gs[0,0], vmin=vmin, vmax=vmax)
        self.modAxes, self.modImage = self.createImageAxes(mod_im, gs[0,1], vmin=vmin, vmax=vmax)

        self.resAxes, self.resImage = self.createImageAxes(res_im, gs[0,2],
                                                           vmin=-res_amp, vmax=res_amp, cmap='RdBu')
        
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
    
    def __init__(self, parent, wxid, name, *args, **kwargs):
        wx.Panel.__init__(self, parent, wxid, *args, **kwargs)

        self.name = name
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
        sizer1.Add(self.label, 0, wx.CENTER, 1)
        sizer1.Add(self.valueText, 0, wx.LEFT, 1)
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2.Add(self.llimitText, 0, wx.LEFT, 1)
        sizer2.Add(self.slider, 1, wx.EXPAND, 1)
        sizer2.Add(self.ulimitText, 0, wx.LEFT, 1)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(sizer1, 1, wx.ALL, 1)
        sizer.Add(sizer2, 1, wx.EXPAND | wx.ALL, 1)
        self.SetSizer(sizer)
        
    
    def init(self, param, name):
        self.name = name        
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
        self.sendChangeEvent()


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
        self.fitButton = wx.Button(self, wx.ID_ANY, 'Fit', (-1, -1), wx.DefaultSize)
        self.saveButton = wx.Button(self, wx.ID_ANY, 'Save', (-1, -1), wx.DefaultSize)

        self.params = {}
        self.X0 = self.addParamCtrl(self.model.x0, 'X_0')
        self.I_e = self.addParamCtrl(self.model.bulge.I_e, 'I_e')
        self.r_e = self.addParamCtrl(self.model.bulge.r_e, 'r_e')
        self.n = self.addParamCtrl(self.model.bulge.n, 'n')
        self.PA_b = self.addParamCtrl(self.model.bulge.PA, 'PA_b')
        self.ell_b = self.addParamCtrl(self.model.bulge.ell, 'ell_b')

        self.Y0 = self.addParamCtrl(self.model.y0, 'Y0')
        self.I_0 = self.addParamCtrl(self.model.disk.I_0, 'I_0')
        self.h = self.addParamCtrl(self.model.disk.h, 'h')
        self.PA_d = self.addParamCtrl(self.model.disk.PA, 'PA_d')
        self.ell_d = self.addParamCtrl(self.model.disk.ell, 'ell_d')
        
        self.resetButton.Bind(wx.EVT_BUTTON, self.onResetButton)
        self.fitButton.Bind(wx.EVT_BUTTON, self.onFitButton)
        self.saveButton.Bind(wx.EVT_BUTTON, self.onSaveButton)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer_l = wx.BoxSizer(wx.VERTICAL)
        sizer_l.Add(self.X0, 1, wx.EXPAND | wx.ALL, 1)
        sizer_l.Add(self.I_e, 1, wx.EXPAND | wx.ALL, 1)
        sizer_l.Add(self.r_e, 1, wx.EXPAND | wx.ALL, 1)
        sizer_l.Add(self.n, 1, wx.EXPAND | wx.ALL, 1)
        sizer_l.Add(self.PA_b, 1, wx.EXPAND | wx.ALL, 1)
        sizer_l.Add(self.ell_b, 1, wx.EXPAND | wx.ALL, 1)

        sizer_r = wx.BoxSizer(wx.VERTICAL)
        sizer_r.Add(self.Y0, 1, wx.EXPAND | wx.ALL, 1)
        sizer_r.Add(self.I_0, 1, wx.EXPAND | wx.ALL, 1)
        sizer_r.Add(self.h, 1, wx.EXPAND | wx.ALL, 1)
        sizer_r.Add(self.PA_d, 1, wx.EXPAND | wx.ALL, 1)
        sizer_r.Add(self.ell_d, 1, wx.EXPAND | wx.ALL, 1)
        
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add(self.resetButton, 0, wx.ALIGN_CENTER | wx.ALL, 1)
        button_sizer.Add(self.fitButton, 0, wx.ALIGN_CENTER | wx.ALL, 1)

        sizer_r.Add(button_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 1)
        sizer_r.Add(self.saveButton, 0, wx.ALIGN_CENTER | wx.ALL, 1)
        sizer.Add(sizer_l, 1, wx.EXPAND | wx.ALL, 1)
        sizer.Add(sizer_r, 1, wx.EXPAND | wx.ALL, 1)

        self.SetSizer(sizer)
        

    def addParamCtrl(self, param, name):
        paramCtrl = ParamCtrl(self, wx.ID_ANY, name)
        self.initParamCtrl(paramCtrl, param, name)
        paramCtrl.Bind(EVT_PARAM_UPDATE_EVENT, self.onParamUpdate)
        return paramCtrl


    def initParamCtrl(self, paramCtrl, param, name):
        paramCtrl.init(param, name)
        self.params[name] = param


    def sendModelUpdateEvent(self, reset=False):
        event = ModelUpdateEvent(model=self.model, reset=reset)
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
        self.resetParams(self.originalModel, reset_plot=True)
        
        
    def resetParams(self, model, reset_plot=False):
        self.model = deepcopy(model)
        self.params = {}
        
        self.initParamCtrl(self.X0, self.model.x0, 'X0')
        self.initParamCtrl(self.Y0, self.model.y0, 'Y0')

        self.initParamCtrl(self.I_e, self.model.bulge.I_e, 'I_e')
        self.initParamCtrl(self.r_e, self.model.bulge.r_e, 'r_e')
        self.initParamCtrl(self.n, self.model.bulge.n, 'n')
        self.initParamCtrl(self.PA_b, self.model.bulge.PA, 'PA_b')
        self.initParamCtrl(self.ell_b, self.model.bulge.ell, 'ell_b')

        self.initParamCtrl(self.I_0, self.model.disk.I_0, 'I_0')
        self.initParamCtrl(self.h, self.model.disk.h, 'h')
        self.initParamCtrl(self.PA_d, self.model.disk.PA, 'PA_d')
        self.initParamCtrl(self.ell_d, self.model.disk.ell, 'ell_d')
        self.sendModelUpdateEvent(reset=reset_plot)

        
    def onFitButton(self, event):
        self.Disable()
        event = FitEvent(model=self.model)
        wx.PostEvent(self.GetEventHandler(), event)
    
    
    def onParamUpdate(self, event):
        param = self.params[event.param]
        param.setValue(event.value)
        try:
            param.setLimits(event.llimit, event.ulimit)
        except:
            logger.warn('Bad limits: %f, %f' % (event.llimit, event.ulimit))
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
        
        model = self.loadModel(model_file)
        
        self.plotPanel = PlotPanel(self)
        self.controlPanel = ControlPanel(self, wx.ID_ANY, model, model_file)

        self.controlPanel.Bind(EVT_MODEL_UPDATE_EVENT, self.onModelUpdate)
        self.controlPanel.Bind(EVT_FIT_EVENT, self.onFit)

        frameSizer = wx.BoxSizer(wx.VERTICAL)
        frameSizer.Add(self.plotPanel, 1, wx.EXPAND)
        frameSizer.Add(self.controlPanel, 0, wx.EXPAND)
        self.SetSizerAndFit(frameSizer)

        self.SetAutoLayout(True)
        self.createMenu()
        
        self.updatePlots(model, reset=True)    
        
    
    
    def createMenu(self):
        menu_bar = wx.MenuBar()
        file_menu = wx.Menu()
        reset_menu_item = file_menu.Append(wx.NewId(), 'Reset', 'Reset the model.')
        fit_menu_item = file_menu.Append(wx.NewId(), 'Fit', 'Fit current model.')
        save_menu_item = file_menu.Append(wx.NewId(), 'Save', 'Save current model.')
        menu_bar.Append(file_menu, 'File')
        self.SetMenuBar(menu_bar)
        
        self.Bind(wx.EVT_MENU, self.controlPanel.onResetButton, reset_menu_item)
        self.Bind(wx.EVT_MENU, self.controlPanel.onFitButton, fit_menu_item)
        self.Bind(wx.EVT_MENU, self.controlPanel.onSaveButton, save_menu_item)
        
        entries = [wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('R'), reset_menu_item.GetId()),
                   wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('F'), fit_menu_item.GetId()),
                   wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('S'), save_menu_item.GetId())]
        acc_table = wx.AcceleratorTable(entries)
        self.SetAcceleratorTable(acc_table)
    
        
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
        
    
    def loadModel(self, model_file):
        if not path.exists(model_file):
            logger.warn('Initial model file not found (%s), guessing one. ' % model_file)
            x0 = (self.flux.shape[1] / 2.0) + 1.0
            y0 = (self.flux.shape[0] / 2.0) + 1.0
            pa, ell = ellipse_params(self.flux, x0, y0)
            r = distance(self.flux.shape, x0, y0, pa, ell)
            r = np.ma.array(r, mask=self.flux.mask)
            hlr = r50(self.flux, r)
            I_e = self.flux.max() * 0.1
            r_e = 0.5 * hlr
            n = 2.0
            I_0 = self.flux.max() * 0.1
            h = 1.0 * hlr
            model = BDModel()
            model.wl = 5635.0
            model.x0.setValue(x0)
            model.x0.setLimitsRel(10, 10)
            model.y0.setValue(y0)
            model.y0.setLimitsRel(10, 10)
            
            model.disk.I_0.setValue(I_0)
            model.disk.I_0.setLimits(0.0, 10.0 * I_0)
            model.disk.h.setValue(h)
            model.disk.h.setLimits(0.0, 5.0 * hlr)
            model.disk.PA.setValue(pa)
            model.disk.PA.setLimits(0.0, 180.0)
            model.disk.ell.setValue(ell)
            model.disk.ell.setLimits(0.0, 1.0)
        
            model.bulge.I_e.setValue(I_e)
            model.bulge.I_e.setLimits(1e-33, 3.0 * I_e)
            model.bulge.r_e.setValue(r_e)
            model.bulge.r_e.setLimits(1e-33, 2.5 * r_e)
            model.bulge.n.setValue(n, vmin=1.0, vmax=5.0)
            model.bulge.PA.setValue(pa)
            model.bulge.PA.setLimits(0.0, 180.0)
            model.bulge.ell.setValue(ell)
            model.bulge.ell.setLimits(0.0, 1.0)
            
            return model
        else:
            return BDModel.load(model_file)
        
        
    def onModelUpdate(self, event):
        self.updatePlots(event.model, event.reset)


    def onFit(self, event):
        wx.BusyCursor()
        model = event.model
        fit_model, converged, chi2 = fit_image(self.flux, self.noise, model, self.psf, mode='NM')
        logger.debug('Fit converged: %s, chi2 = %f' % (converged, chi2))
        self.controlPanel.resetParams(fit_model, reset_plot=False)
        self.updatePlots(fit_model)
        self.controlPanel.Enable()

################################################################################
    
