#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QDSpy module - creates and manages the presentation window

'View'
  A class to create and manage the stimulus presentation window and, if
  needed, a smaller stimulus window on the user screen to allow the user
  to follow stimulus presentation in full-screen multi-monitor mode.
  This class is a graphics API independent.

Copyright (c) 2013-2019 Thomas Euler
All rights reserved.
"""
# ---------------------------------------------------------------------
__author__ 	= "code@eulerlab.de"

import QDSpy_global as glo
import QDSpy_stim_support as ssp
import Graphics.renderer_opengl as rdr

# ---------------------------------------------------------------------
# Adjust global parameters depending on command line arguments
#
'''
global QDSpy_graphicsAPI
QDSpy_graphicsAPI = cfg.getParsedArgv().timing
if QDSpy_graphicsAPI == 0:
  import QDSpy_core_GL_default as grx
"""
elif QDSpy_graphicsAPI == 1:
  import  QDSpy_core_GL_alter1  as grx
elif QDSpy_graphicsAPI == 2:
  import  QDSpy_core_GL_alter2  as grx
"""
'''
# =====================================================================
#
# ---------------------------------------------------------------------
class View:
  """ Creates and manages the stimulus presentation window(s)
  """
  def __init__(self, _Stage, _Conf):
    # Initializing
    #
    self.Stage    = _Stage
    self.Conf     = _Conf
    self.Renderer = rdr.Renderer(self, glo.QDSpy_KEY_KillPresent)
    self.__reset()

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  def __reset(self):
    # Resets internal states; don't use
    #
    self.onKeyboardProc = None
    self.onDrawProc     = None
    self.isWinAvailable = False
    # ...

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  def createStimulusWindow(self):
    # Create window(s) ...
    #
    # Log some information about OpenGL on this machine
    #
    ssp.Log.write("INFO", self.Renderer.get_info_GL_str())
    ssp.Log.write("INFO", self.Renderer.get_info_GLSL_str())
    ssp.Log.write("INFO", self.Renderer.get_info_renderer_str())
    ssp.Log.write("INFO", self.Renderer.get_implementation_str())
    ssp.Log.write("INFO", "Expected   : {0:.1f} Hz"
                          .format(self.Stage.scrReqFreq_Hz))

    # Check if window, overlay or fullscreen mode is requested
    #
    nScr           = self.Renderer.get_screen_count()
    winXCorrFact   = 1.0
    self.isFullScr = (self.Stage.dxScr <= 0) or (self.Stage.dyScr <= 0)
    self.winTitle  = glo.QDSpy_fullScrWinName

    self.iScr      = self.Stage.scrIndex
    assert (self.iScr < nScr), "Screen index issue in `createStimulusWindow`"

    if self.Stage.useScrOvl:
      # Overlay mode - Wide window for two overlain displays
      #
      xy            = (self.Stage.offXScr1_pix, self.Stage.offYScr1_pix)
      dxy           = (self.Stage.dxScr12, self.Stage.dyScr12)
      self.winPre   = self.Renderer.create_window(self.iScr, self.winTitle,
                                                  _dx=dxy[0], _dy=dxy[1],
                                                  _isScrOvl=True,
                                                  _iScrGUI=self.Stage.scrIndexGUI,
                                                  _offset=xy)
      ssp.Log.write("INFO", self.Renderer.get_info_screen_str())
      ssp.Log.write("ok", "Overlay mode, 2x {0}x{1} pixels, starting on "
                          "screen #{2}".format(dxy[0]//2, dxy[1], self.iScr))

    elif self.isFullScr:
      # Fullscreen mode - Normal full-screen window for a single display
      #
      dxy           = self.Renderer.get_screen_size(self.iScr)
      self.winPre   = self.Renderer.create_window(self.iScr, self.winTitle)
      ssp.Log.write("INFO", self.Renderer.get_info_screen_str())
      ssp.Log.write("ok", "Fullscreen mode, {0}x{1} pixels, on screen #{2}"
                          .format(dxy[0], dxy[1], self.iScr))

      self.winPre.set_mouse_visible(False)

      if self.Conf.useCtrlWin:
        div = int(1/self.Conf.ctrlWinScale)
        self.winPreview = self.Renderer.create_window(1, "", dxy[0]//div,
                                                      dxy[1]//div, 50,50,
                                                      self.Conf.ctrlWinScale)

    else:
      # Window mode
      #
      xy            = (self.Stage.xWinLeft, self.Stage.yWinTop)
      dxy           = (self.Stage.dxScr, self.Stage.dyScr)
      self.winTitle = glo.QDSpy_versionStr
      self.winPre   = self.Renderer.create_window(self.iScr, self.winTitle,
                                                  dxy[0], dxy[1], xy[0], xy[1])
      ssp.Log.write("ok", "Window mode, {0}x{1} pixels".format(dxy[0], dxy[1]))

      # Adjust scaling factor such that presentation in window is
      # to scale; assuming that 1 pix = 1um is true for the current
      # screen's maximal resolution
      #
      """
      self.screens  = grx.getScreens()
      winXCorrFact  = float(self.winWidth) /self.screens[0].width
      """

    # Update self and stage object
    #
    self.winPreWidth        = dxy[0]
    self.winPreHeight       = dxy[1]
    self.Stage.dxScr        = dxy[0]
    self.Stage.dyScr        = dxy[1]
    self.Stage.isFullScr    = self.isFullScr
    self.Stage.winXCorrFact = winXCorrFact

    # Try to force vsync, if requested
    #
    result = self.Renderer.force_vSync()
    if result < 0:
      ssp.Log.write("WARNING", "SwapIntervalEXT() not supported")
    elif result >= 0:
      ssp.Log.write(" ", "{0:11}: forced fsync"
                    .format("ENABLED" if self.Conf.fSync else "disabled"))
      if result == 1:
        ssp.Log.write("ok", "SwapIntervalEXT() reported success")

    # Success ...
    #
    self.isWinAvailable = True

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  def clear(self, _RGB=[]):
    self.Renderer.clear_windows(_RGB)

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  def present(self):
    self.Renderer.present()

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  def dispatch_events(self):
    self.Renderer.dispatch_events()

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  def createBatch(self, _isScrOvl=False):
    return rdr.Batch(_isScrOvl)

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  def setOnKeyboardHandler(self, _onKeybProc):
    if self.isWinAvailable:
      self.onKeyboardProc = _onKeybProc

  def setOnDrawHandler(self, _onDrawProc):
    if self.isWinAvailable:
      self.onDrawProc     = _onDrawProc

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  def onKeyboard(self, _key, _x, _y):
    if not self.onKeyboardProc is None:
      self.onKeyboardProc(_key, _x, _y)

  def onDraw(self):
    if not self.onDrawProc is None:
      self.onDrawProc()

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  def startRenderingLoop(self, _Pre):
    """ Start main rendering loop
    """
    self.Renderer.start_main_loop(_Pre)

  def killWindows(self):
    """ Kill OpenGL window(s) ...
    """
    if self.isWinAvailable:
      # Exit main loop, which kills the window(s)
      #
      self.Renderer.end_main_loop()
      self.__reset()

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  def prepareGrabStim(self):
    # Prepare grabbing the stimulus window
    #
    self.Renderer.prepare_record_win(0)
    ssp.Log.write("DEBUG", "Not implemented: Renderer.prepare_record_win")

  def grabStimFrame(self):
    # Grab the current frame of the stimulus window
    #
    self.Renderer.grab_frame()
    ssp.Log.write("DEBUG", "Not implemented: Renderer.grab_frame")


# ---------------------------------------------------------------------
