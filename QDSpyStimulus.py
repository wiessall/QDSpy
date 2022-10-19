# -*- coding: utf-8 -*-
"""
Top level classes to create stimuli with QDSpy.

@author: Tom Boissonnet
"""

import QDS
import abc
import numpy as np
import math
from scipy.signal import convolve2d
import glob

def degree_to_pixel(degrees, screen_distance, pixel_size):
    if type(degrees) is not list:
        degrees = [degrees]
    return [math.tan(math.radians(degree)) * screen_distance // pixel_size for degree in degrees]
    
def pixel_to_degree(pixels, screen_distance, pixel_size):
    if type(pixels) is not list:
        pixels = [pixels]
    return [math.degrees(math.atan(px*pixel_size/screen_distance)) for px in pixels]
    
def um_on_retina_to_degree(ums):
    if type(ums) is not list:
        ums = [ums]
    return [um * 0.032 for um in ums]
    
def um_on_retina_to_pixel(ums, pixel_size_mm):
    if type(ums) is not list:
        ums = [ums]
    return [int(um * pixel_size_mm) for um in ums]

class QDSpy_Stimulus():

    BARCODE_LEN = 4
    __metaclass__ = abc.ABCMeta

    def __init__(self, _sName, _sDescription, 
                width, height, barcode, optimize=False, 
                IHigh=255, ILow=0, IHalf=127, 
                red_FF=False, red_cycle=False, reds=[50,100,150,200,255], mag_red=(1,1)):
        """ The base class to define stimuli. This is an abstract class, please use a subclass such a Array_Stimulus or Fullfield_Stimulus.

        Parameters
        ----------
        _sName
            The name of the stimulus
        _sDescription
            The description of the stimulus
        width
            The width of the stimulus in pixels
        height
            The height of the stimulus in pixels
        barcode
            The stimulus barcode. A string of 4 integers between 1 and 4.
        optimize
            Flag for Array_Stimulus to optimize the number of object displayed. If True, a function will
            replace smaller squares with same intensity placed together by a single bigger square. This
            works only for stimuli where the screen is filled with black and white squares.
            
            
        unit
            The unit in which the size of the shapes in the stimulus are expressed. Can be 
            one of 'pixel', 'degree' or 'um_on_retina'            
        screen_distance_mm
            The distance of the retina to the screen in millimeter. Not used in the case the unit is um_on_retina
        pixel_size_mm
            The size of a pixel (width or heigh) in millimeter.
            
            
        IHigh
            The maximum value of a color for the stimulus.
        ILow
            The minimum value of a color for the stimulus.
        IHalf
            The half value of a color for the stimulus.
            

        red_FF
            Flag to indicate if the red signals are used on the full screen.
        red_cycle:
            In the case the red_FF flag is True, the red cycle will use more than two
            intensities to encode the red signals
        reds:
            The values used for the red signals in the case of a red_FF
        """

        if len(barcode) != 4 and barcode%10 <= 4 and barcode%100 <= 4 and barcode%1000 <= 4 and barcode%10000 <= 4:
            raise Exception("The barcode should be compose of 4 numbers between 1 and 4")
            
        #### INITIALIZATION OF MAIN PARAMETERS OF THE STIMULUS ####
        self.frame_time   = 1/60
        self.clear_time   = 6/60 #Clear time used before and at the end of the stimulus presentation, to heat up the display
        self.bkg_width    = 5000
        self.p            = {}
        self.p["_sName"]  = _sName
        self.p["_sDescr"] = _sDescription
        self.p["width"]   = width
        self.p["height"]  = height
        self.p["barcode"] = barcode
        self.optimize = optimize

        #### INITIALIZATION OF LIGTH INTENSITIES PARAMETERS OF THE STIMULUS ####
        self.p["IHigh"]   = IHigh
        self.p["ILow"]    = ILow
        self.IHalf        = IHalf

        #### INITIALIZATION OF RED SIGNALS PARAMETERS OF THE STIMULUS ####
        self._red_index    = 1000000
        self.bkg_idx       = -10
        self._marker_idx   = -11
        self.red_FF       = red_FF
        self.red_cycle    = red_cycle
        self.mag_red      = mag_red
        self._marker_base_size = 200
              
        if not self.red_FF and not self.red_cycle:
            reds = [0]
        elif not self.red_cycle:
            reds = [reds[0], reds[-1]]            
        self.reds_mark = reds
        self.reds_obj  = self.reds_mark
        if self.red_cycle and not self.red_FF:
            self.reds_obj = [0]
            
        self.rgbs_high   = [(red, IHigh, IHigh) for red in self.reds_obj]
        self.rgbs_low    = [(red, ILow,  ILow)  for red in self.reds_obj]
        self.rgbs_half   = [(red, IHalf, IHalf) for red in self.reds_obj]
        self.rgbs_blue   = [(red, ILow,  IHigh) for red in self.reds_obj]
        self.rgbs_green  = [(red, IHigh, ILow)  for red in self.reds_obj]
        self.red_obj_idx  = [(i*self._red_index) for i in range(len(self.reds_obj))]
        self.red_mark_idx = [(i*self._marker_idx)+self._marker_idx for i in range(len(self.reds_mark))]
        self.marker_count, self.marker_max = 4, len(self.reds_mark)    #Start marker at 4 so the first frame is not a zero but a high signal -> we can check drop of first frame.      

    def start_script(self):
        QDS.StartScript()
        self.__signature()

    def end_script(self):
        self.scene_clear(self.clear_time, 1)
        self.scene_clear(self.clear_time, 0)
        QDS.EndScript()

    def scene_clear(self, duration, marker_on):
        if not self.red_FF and not self.red_cycle:
            QDS.Scene_Clear(duration, marker_on)
            
        elif not self.red_FF and self.red_cycle:
            if marker_on:
                QDS.Scene_RenderEx(duration,[self.bkg_idx, self.red_mark_idx[-1]], [(0,0),(1280//2,-720//2)], [(1,1),self.mag_red], [0, 0], 0)
            else:
                QDS.Scene_RenderEx(duration,[self.bkg_idx, self.red_mark_idx[0]] , [(0,0),(1280//2,-720//2)], [(1,1),self.mag_red], [0, 0], 0)
                
        else:
            if marker_on:
                QDS.Scene_RenderEx(duration,[self.bkg_idx+self.red_obj_idx[-1]], [(0,0)], [(1,1)], [0], 0)
            else:
                QDS.Scene_RenderEx(duration,[self.bkg_idx+self.red_obj_idx[0]] , [(0,0)], [(1,1)], [0], 0)


    def __signature(self):
        self.scene_clear(self.clear_time, 0)

        self.scene_clear(self.frame_time,1)
        self.scene_clear(self.frame_time,0)
        for repl in map(int,self.p["barcode"]): #map with int change all number chars to individual integers
            for i in range(repl):
                self.scene_clear(self.frame_time,1)
            self.scene_clear(self.frame_time,0)
        self.scene_clear(self.frame_time,1)

        self.scene_clear(self.clear_time, 0)

    def log_user_parameters(self):
        QDS.LogUserParameters(self.p)

    def initialize(self):
        QDS.Initialize(self.p['_sName'],self.p['_sDescr'])


    @abc.abstractmethod
    def scene_render(self,duration,iobjs,opos = (0,0),marker_on = 0):
        """Method to render all the objects of the frame"""

    def define_objects(self):        
        """Define the objects used by the script"""
        #Objects with FF red to replace the scene clear
        indexes = []
        for i, red_idx in enumerate(self.red_obj_idx):
            QDS.DefObj_BoxEx(self.bkg_idx + red_idx, self.bkg_width, self.bkg_width)
            indexes.append(  self.bkg_idx + red_idx)

        QDS.SetObjColorEx(indexes, self.rgbs_half)
            
        if self.red_cycle and not self.red_FF:    
            for i, red_idx in enumerate(self.red_mark_idx):                
                QDS.DefObj_BoxEx(red_idx, self._marker_base_size, self._marker_base_size)
                QDS.SetObjColorEx([red_idx], [(self.reds_mark[i], self.p["ILow"], self.p["ILow"])])
    
    def loop(self, n_repeats, func):
        QDS.Loop(n_repeats,func)
        
    def set_bkg_color(self, color):
        QDS.SetBkgColor(color)
        
    def define_additional_objects(self, indexes, colors, dimensions=None, shapes=None):
        if len(indexes) != len(colors):
            raise Exception("Indexes and colors lenght is different. Data not understood")
            
        if dimensions is None:
            dimensions = [(self.p["width"],self.p["height"])] * len(indexes)
            
        if shapes is None:
            shapes = [self.stim_shape]*len(indexes)

        color_list = []    
        index_list = []
        for m, red_idx in enumerate(self.red_obj_idx):
            for j, idx in enumerate(indexes):

                if shapes[j]=="box":
                    QDS.DefObj_BoxEx(    idx+red_idx,dimensions[j][0],dimensions[j][1])
                if shapes[j]=="ellipse":
                    QDS.DefObj_EllipseEx(idx+red_idx,dimensions[j][0],dimensions[j][1])
                    
                color_list.append((self.reds_obj[m],colors[j][1],colors[j][2]))
                index_list.append(idx+red_idx)
            if not self.red_FF and not self.red_cycle:
                break

        QDS.SetObjColorEx(index_list, color_list)
        
    def define_additional_objects_with_shader(self, indexes, shader_names, shader_parameters, dimensions=None, shapes=None):
           
        if dimensions is None:
            dimensions = [(self.p["width"],self.p["height"])] * len(indexes)
        if shapes is None:
            shapes = [self.stim_shape]*len(indexes)
            
        color_list = []    
        index_list = []
        shader_list= []
        for m, red_idx in enumerate(self.red_obj_idx):
            for j, idx in enumerate(indexes):

                if shapes[j]=="box":
                    QDS.DefObj_BoxEx(    idx+red_idx,dimensions[j][0],dimensions[j][1], _enShader=1)
                elif shapes[j]=="ellipse":
                    QDS.DefObj_EllipseEx(idx+red_idx,dimensions[j][0],dimensions[j][1], _enShader=1)
                
                QDS.DefShader(idx+red_idx, shader_names[j])
                QDS.SetShaderParams(idx+red_idx, shader_parameters[j])
                shader_list.append(idx+red_idx)
                
            if not self.red_FF and not self.red_cycle:
                break
                    
        QDS.SetObjShader(shader_list, shader_list)
        
    def _scene_render(self, duration, iobjs, opos, mag, angle, marker_on):
        red_idx = 0
        if self.red_FF: # We get the red idx of the objects -> All of them must display the correct red intensity.
            if self.red_cycle:
                red_idx = self.red_obj_idx[self.marker_count]
                self.marker_count = (self.marker_count+1) % self.marker_max
            elif marker_on:
                red_idx = self.red_obj_idx[-1]
            else:
                red_idx = self.red_obj_idx[0]
            marker_on   = 0                         # Set to zero to prevent QDSpy to put his own red marker.
            iobjs       = list(np.array(iobjs) + red_idx)  
                
        elif self.red_cycle:
            mark_idx = self.red_mark_idx[self.marker_count]
            self.marker_count = (self.marker_count+1) % self.marker_max
            marker_on   = 0                         # Set to zero to prevent QDSpy to put his own red marker.
            iobjs       = iobjs + [mark_idx] # We add the marker at the bottom right of the screen (same standards as QDSpy)
            opos        = opos  + [(1280//2,-720//2)]
            mag         = mag   + [self.mag_red]
            angle       = angle + [0]
        
        
        #Adding a background    
        iobjs       = [self.bkg_idx + red_idx] + iobjs
        opos        = [(0,0)]                  + opos 
        mag         = [(1,1)]                  + mag
        angle       = [0]                      + angle
        QDS.Scene_RenderEx(duration, iobjs, opos, mag, angle, marker_on)    
        
class Fullfield_Stimulus(QDSpy_Stimulus):

    def __init__(self ,stim_shape,*argv, **kwarg):
        """ Define a stimulus of type "fullfield": the screen filled by a single object.
        
        Parameters
        ----------
        stim_shape
            The shape of the stimulus, either "box" or "ellipse".
        """
        super().__init__(*argv, **kwarg)
        self.stim_shape = stim_shape

    def define_objects(self):
        """For the fullfield stimuli, I use 281 different objects:
            IDs 0->255:
                Range of grey stimuli, intensity equal id

            IDs 1000 to 1044:
                Set of mixed colors, encoded on two digits e.g 1042 means
                intensity 4 for green and 2 for blue.
                Intensities 0 to 4: 0,63,127,191,255
        """
        super().define_objects()        
        
        if self.stim_shape not in ["box","ellipse"]:
            raise Exception("Stim shape must be 'box' or 'ellipse'")
        
        color_list=[]
        index_list = []
        for m, red_idx in enumerate(self.red_obj_idx):
            for i in range(256): #Defining Gray levels (0->255)
                color_list.append((self.reds_obj[m],i,i))
                index_list.append(i+red_idx)
                if self.stim_shape=="box":
                    QDS.DefObj_BoxEx(i+red_idx,self.p["width"],self.p["height"])
                elif self.stim_shape=="ellipse":
                    QDS.DefObj_EllipseEx(i+red_idx,self.p["width"],self.p["height"])
            for i in range(5):   #Defining Color mixes
                for j in range(5):
                
                    uv, blue = i*(self.p["IHigh"]-self.p["ILow"])//4 + self.p["ILow"], j*(self.p["IHigh"]-self.p["ILow"])//4 + self.p["ILow"]
                    if uv == 256:
                        uv   = 255
                    if blue == 256:
                        blue = 255
                    color_list.append((self.reds_obj[m], uv, blue))
                    index_list.append(int("10"+str(i)+str(j))+red_idx)
                    
                    if self.stim_shape=="box":
                        QDS.DefObj_BoxEx(int(("10"+str(i)+str(j)))+red_idx,self.p["width"],self.p["height"])
                    elif self.stim_shape=="ellipse":
                        QDS.DefObj_EllipseEx(int(("10"+str(i)+str(j)))+red_idx,self.p["width"],self.p["height"])
                        
            if not self.red_FF and not self.red_cycle: 
                break #Then we break because we don't need all red colors
        
        QDS.SetObjColorEx(index_list, color_list)

    def scene_render(self,duration,iobjs,opos = [(0,0)],marker_on = 0, mag = [(1,1)], angle = [0]):
        # Adapt the objs id, mag and angle lists
        if type(iobjs) is int:
            iobjs = [iobjs]
        if len(mag)   == 1:
            mag   = mag*len(iobjs)
        if len(angle) == 1:
            angle = angle*len(iobjs)
            
        self._scene_render(duration, iobjs, opos, mag, angle, marker_on)           
           
class Array_Stimulus(QDSpy_Stimulus):
    def __init__(self, dx, dy, unit, screen_distance_mm, pixel_size_mm,*argv, **kwarg):
        """ Define a stimulus of type "array": the screen is divided in pixels of a certain size.
        
        Parameters
        ----------
        dx
            The width the screen is divided in. The unit must be the same as the parameter "unit"
        dy
            The height the screen is divided in. The unit must be the same as the parameter "unit"
        """
        super().__init__(*argv,**kwarg)
        #### INITIALIZATION OF UNITS PARAMETERS OF THE STIMULUS ####
        if unit not in ["pixel", "degree", "um_on_retina"]:
            raise Exception("The unit to define object sizes does not match a known unit. It must be one of [pixel, degree, um_on_retina]")
        self.unit               = unit
        self.screen_distance_mm = screen_distance_mm
        self.pixel_size_mm      = pixel_size_mm    
        
        if self.unit == "pixel":
            dx_px = dx
            dy_px = dy
        elif self.unit == "degree":
            dx_px = degree_to_pixel(dx, self.screen_distance_mm, self.pixel_size_mm)
            dy_px = degree_to_pixel(dx, self.screen_distance_mm, self.pixel_size_mm)
        elif self.unit == "um_on_retina":
            dx_px  = um_on_retina_to_pixel(dx, self.pixel_size_mm)
            dy_px  = um_on_retina_to_pixel(dy, self.pixel_size_mm)
            dx_deg = um_on_retina_to_degree(dx)
            dy_deg = um_on_retina_to_degree(dy)
        
        if not self.unit == "um_on_retina":        
            dx_deg = pixel_to_degree(dx_px, self.screen_distance_mm, self.pixel_size_mm)
            dy_deg = pixel_to_degree(dy_px, self.screen_distance_mm, self.pixel_size_mm)
        
        if dx_px <= 0 or dy_px <= 0:
            raise Exception("The pixel size of the box for ", self.p["_sName"], "stimulus is lesser or equal to 0.")
        
        self.p["dx_deg"] = dx_deg
        self.p["dy_deg"] = dy_deg
        self.p["dx_px"]  = dx_px
        self.p["dy_px"]  = dy_px
        self.n_xBox = self.p["width"] // dx_px
        self.n_yBox = self.p["height"]// dy_px

    def define_objects(self):
        super().define_objects()
        dx,dy = self.p["dx_px"],self.p["dy_px"]
        x_center = (self.p["width"]/2)-(dx/2)
        y_center = (self.p["height"]/2)-(dy/2)
        self.pos_list = [(x*dx -x_center,y*dy -y_center) for y in range(self.n_yBox)for x in range(self.n_xBox)]*2

        color_list = []
        index_list = []

        for m, red_idx in enumerate(self.red_obj_idx):
            QDS.DefObj_BoxEx(-1 + red_idx,self.p["width"],self.p["height"])
            color_list.append(self.rgbs_low[m])
            index_list.append(-1 + red_idx)
            
            QDS.DefObj_BoxEx(128 + red_idx,self.p["width"],self.p["height"])
            color_list.append(self.rgbs_half[m])
            index_list.append(128 + red_idx)
        
            QDS.DefObj_BoxEx(0 + red_idx, self.p["dx_px"], self.p["dy_px"])
            color_list.append(self.rgbs_low[m])
            index_list.append(0 + red_idx)
            
            QDS.DefObj_BoxEx(1 + red_idx, self.p["dx_px"], self.p["dy_px"])
            color_list.append(self.rgbs_high[m])
            index_list.append(1 + red_idx)
            for w in ['1','2','3','4']:
                for h in ['1','2','3','4']:
                    QDS.DefObj_BoxEx(int(w+h) + red_idx, self.p["dx_px"] * int(w), self.p["dy_px"] * int(h))
                    color_list.append(self.rgbs_high[m])
                    index_list.append(int(w+h) + red_idx)  
            
            if not self.red_FF:
                break  #After the first loop, all the object we will use are defined.
                
        QDS.SetObjColorEx(index_list, color_list)                        


    def scene_render(self, duration, iobjs=[0], opos = [(0,0)], iobjs_black=[], iobjs_white=[], marker_on = 0):
        if self.optimize:
            new_opos, new_iobjs = self.__optimize_display(iobjs_white)
            opos    = [(0,0)] + new_opos #Adding a black background (size of the Array stimulus)
            iobjs   = [-1]    + new_iobjs
        else:
            pos_white = [self.pos_list[i] for i in iobjs_white]
            pos_black = [self.pos_list[i] for i in iobjs_black]
            opos      = opos + pos_black + pos_white 
            iobjs     = iobjs + len(pos_black) * [0] + len(pos_white) * [1]
        n_obj = len(opos)
        mag   = [(1,1)] * n_obj  
        angle = [0]     * n_obj 
        self._scene_render(duration, iobjs, opos, mag, angle, marker_on)   
        
    def __optimize_display(self, indexes):
        new_indexes = []
        new_pos = []
        scene = np.zeros(self.n_xBox*self.n_yBox,dtype=int)
        scene[indexes]=1
        scene = scene.reshape(self.n_yBox,self.n_xBox)
        
        for w,h in [(4,4),(4,3),(3,4),(3,3),(4,2),(2,4),(2,3),(3,2),(2,2),(4,1),(1,4),(3,1),(1,3),(2,1),(1,2)]:
            if self.n_xBox < w:
                continue
            if self.n_yBox < h:
                continue    
            x_center = self.p['width'] /2 - (self.p["dx_px"] * w /2)
            y_center = self.p['height']/2 - (self.p["dy_px"] * h /2)
            kernel   = np.ones((h, w))
            conv_mat = convolve2d(scene, kernel, mode='same') == np.sum(kernel)
            idx_mat  = np.where(conv_mat == 1)
            tmp_pos  = list(zip(((idx_mat[1]-(w//2))*self.p["dx_px"])-x_center,((idx_mat[0]-(h//2))*self.p["dy_px"])-y_center))
            
            new_indexes.extend([int(str(w)+str(h))]* len(tmp_pos))            
            new_pos.extend(tmp_pos)
            
            for w_range in range(    -(w//2), (w//2)+(w%2)): #Replace the initial image by 0 where bigger shape has been found. Can it be done with better indexing?
                for h_range in range(-(h//2), (h//2)+(h%2)):
                    scene[idx_mat[0]+h_range, idx_mat[1]+w_range] = 0
            
        isolated_pos = [self.pos_list[i] for i in np.flatnonzero(scene)]
        new_pos.extend(isolated_pos)
        new_indexes.extend([1]*len(isolated_pos))

        return new_pos, new_indexes


class MovingGrating_Stimulus(QDSpy_Stimulus):
    def __init__(self ,stim_shape, spatial_freqs, drift_speeds, unit, screen_distance_mm, pixel_size_mm, *argv, **kwarg):
        """ Define a stimulus of type "moving grating": the screen is filled by a single grating moving.
        
        Parameters
        ----------
        stim_shape
            The shape of the stimulus, either "box" or "ellipse".
        spatial_freqs
            A list of spatial frequencies for the stimulus.
        drift_speeds
            A list of drift speeds for the stimulus.
        """
        super().__init__(*argv, **kwarg)
        #### INITIALIZATION OF UNITS PARAMETERS OF THE STIMULUS ####
        if unit not in ["pixel", "degree", "um_on_retina"]:
            raise Exception("The unit to define object sizes does not match a known unit. It must be one of [pixel, degree, um_on_retina]")
        self.unit               = unit
        self.screen_distance_mm = screen_distance_mm
        self.pixel_size_mm      = pixel_size_mm
        
        if self.unit == "pixel":
            spatial_freqs_px  = spatial_freqs
            drift_speeds_px   = drift_speeds            
        elif self.unit == "degree":
            spatial_freqs_px  = degree_to_pixel(spatial_freqs, self.screen_distance_mm, self.pixel_size_mm)
            drift_speeds_px   = degree_to_pixel(drift_speeds, self.screen_distance_mm, self.pixel_size_mm)
        elif self.unit == "um_on_retina":
            spatial_freqs_px  = um_on_retina_to_pixel(spatial_freqs, self.pixel_size_mm)
            drift_speeds_px   = um_on_retina_to_pixel(drift_speeds, self.pixel_size_mm)
            spatial_freqs_deg = um_on_retina_to_degree(spatial_freqs)
            drift_speeds_deg  = um_on_retina_to_degree(drift_speeds)
        
        if not self.unit == "um_on_retina":
            spatial_freqs_deg = pixel_to_degree(spatial_freqs_px, self.screen_distance_mm, self.pixel_size_mm)
            drift_speeds_deg  = pixel_to_degree(drift_speeds_px, self.screen_distance_mm, self.pixel_size_mm)
        
        self.p["spatial_freqs_deg"] = spatial_freqs_deg
        self.p["drift_speeds_deg"] = drift_speeds_deg     
        self.p["spatial_freqs_px"] = spatial_freqs_px
        self.p["drift_speeds_px"] = drift_speeds_px
        self.stim_shape = stim_shape
        self.spatial_freqs = spatial_freqs_px
        self.drift_speeds = drift_speeds_px
        self.n_grat = len(self.spatial_freqs) * len(self.drift_speeds)

    def define_objects(self):
        super().define_objects()

        if self.stim_shape not in ["box","ellipse"]:
            raise Exception("Stim shape must be 'box' or 'ellipse'")
        
        color_list  = []
        index_list  = []
        shader_list = []
        for m, red_idx in enumerate(self.red_obj_idx):
            if self.stim_shape=="box":
                QDS.DefObj_BoxEx(    -1+red_idx, self.p["width"], self.p["height"])
            elif self.stim_shape=="ellipse":
                QDS.DefObj_EllipseEx(-1+red_idx, self.p["width"], self.p["height"])
            index_list.append(-1+red_idx)
            color_list.append(self.rgbs_half[m])
            for i in range(self.n_grat):
                if self.stim_shape=="box":
                    QDS.DefObj_BoxEx(    i+red_idx, self.p["width"], self.p["height"], _enShader=1)
                elif self.stim_shape=="ellipse":
                    QDS.DefObj_EllipseEx(i+red_idx, self.p["width"], self.p["height"], _enShader=1)
                index_list.append(i+red_idx)
                color_list.append(self.rgbs_high[m])

            for i, spat_freq in enumerate(self.spatial_freqs):
                for j, speed in enumerate(self.drift_speeds):
                    indx = j+i*len(self.drift_speeds)
                    n_cycle_s = spat_freq/speed
                    QDS.DefShader(indx+red_idx,"SQUARE_WAVE_GRATING")
                    QDS.SetShaderParams(indx+red_idx, [spat_freq, n_cycle_s, (self.reds_obj[m], self.p["ILow"],self.p["ILow"], 255), (self.reds_obj[m],self.p["IHigh"],self.p["IHigh"], 255) ])
                    shader_list.append( indx+red_idx)
                    
            if not self.red_FF:
                break
        
        """ # To be implemented later -> Bug solving ongoing
        if self.red_cycle and not self.red_FF: 
            self._marker_idx = 8949651
            self.red_mark_idx = [(i*self._marker_idx)+self._marker_idx for i in range(len(self.reds_mark))]
            for i, red_idx in enumerate(self.red_mark_idx):                
                QDS.DefObj_BoxEx(   red_idx, self._marker_base_size, self._marker_base_size, _enShader=1)
                QDS.DefShader(      red_idx,"SQUARE_WAVE_GRATING")
                QDS.SetShaderParams(red_idx, [spat_freq, n_cycle_s, (self.reds_mark[i], self.p["ILow"],self.p["ILow"], 255), (self.reds_mark[i],self.p["ILow"],self.p["ILow"], 255) ])
                shader_list.append( red_idx)
        """
        QDS.SetObjShader(shader_list, shader_list)
        QDS.SetObjColorEx(index_list,color_list)

    def scene_render(self,duration,iobjs,opos = [(0,0)],marker_on = 0, mag = [(1,1)], angle = [0]):
        if type(iobjs) is int:
            iobjs = [iobjs]
        if len(mag)   == 1:
            mag   = mag*len(iobjs)
        if len(angle) == 1:
            angle = angle*len(iobjs)
            
        self._scene_render(duration, iobjs, opos, mag, angle, marker_on)  

class Video_Stimulus(QDSpy_Stimulus):

    def __init__(self ,video_name, *argv, **kwarg):
        """ Define a stimulus of type "fullfield": the screen filled by a single object.
        
        Parameters
        ----------
        video_name
            The name of the video to load.
        """
        super().__init__(*argv, **kwarg)
        self.video_name = video_name

    def define_objects(self):
        """For the video stimuli, a single object is defined, with id 1.
        """
        super().define_objects()
        QDS.DefObj_Video(1, self.video_name) 

    def scene_render(self, duration, iobj, opos = (0,0), marker_len = 1, mag = (1,1), angle = 0, alpha = 255):
        # Adapt the objs id, mag and angle lists
        QDS.Start_Video(iobj, opos, mag, alpha, angle) 
        for i in range(duration//marker_len):
            self.scene_clear(marker_len, i%2==0)
            #QDS.Scene_Clear(marker_len, i%2==0)


class Fear_Stimulus(QDSpy_Stimulus):

    def __init__(self, *argv, **kwarg):
        """ Define a stimulus of type "fear": the screen filled by a controllable ellipse in front of a controllable
        background.

        Parameters
        ----------
        """
        super().__init__(*argv, **kwarg)

    def define_objects(self):
        """For the fullfield stimuli, I use 281 different objects:
            IDs 0->255:
                Range of grey background, intensity equal id

            IDs 1000 to 1255:
                Range of grey ellipse, intensity equal id
        """

        super().define_objects()

        color_list = []
        index_list = []
        for m, red_idx in enumerate(self.red_obj_idx):
            for i in range(256):  # Defining Gray levels (0->255)
                color_list.append((self.reds_obj[m], i, i))
                color_list.append((self.reds_obj[m], i, i))
                index_list.append(i + red_idx)
                index_list.append(int("10" + "{0:0=3d}".format(i)) + red_idx)
                QDS.DefObj_BoxEx(i + red_idx, self.bkg_width, self.bkg_width)
                QDS.DefObj_EllipseEx(int("10" + "{0:0=3d}".format(i)) + red_idx, self.p["width"], self.p["height"])
            # index_list.insert(0, 2000)
            # color_list.insert(0, (self.reds_obj[m], i, i))
            # QDS.DefObj_EllipseEx(2000, self.p["width"], self.p["height"])

            if not self.red_FF and not self.red_cycle:
                break  # Then we break because we don't need all red colors
        self.color = color_list
        self.index = index_list
        QDS.SetObjColorEx(index_list, color_list)
        

    def scene_render(self, duration, iobjs, opos=[(0, 0)], marker_on=0, mag=[(1, 1)], angle=[0]):
        # Adapt the objs id, mag and angle lists
        if type(iobjs) is int:
            iobjs = [iobjs]
        if len(mag) == 1:
            mag = mag * len(iobjs)
        if len(angle) == 1:
            angle = angle * len(iobjs)

        self._scene_render(duration, iobjs, opos, mag, angle, marker_on)

""" NOT SUPPORTED AT THE MOMENT
class Ring_Stimulus(QDSpy_Stimulus):
    def __init__(self, cell_type, *argv, **kwarg):
        super().__init__(*argv, **kwarg)
        self.cell_type = cell_type

    def define_objects(self):
        super().define_objects()
        QDS.DefObj_EllipseEx(0, self.p["width"],self.p["height"])
        QDS.DefObj_EllipseEx(1, self.p["width"],self.p["height"])
        QDS.DefObj_EllipseEx(2, self.p["width"],self.p["height"])
        QDS.DefObj_BoxEx(-1, 5000,5000)
        if self.red_FF:
            QDS.DefObj_EllipseEx(0+self.red_index, self.p["width"],self.p["height"])
            QDS.DefObj_EllipseEx(1+self.red_index, self.p["width"],self.p["height"])
            QDS.DefObj_EllipseEx(2+self.red_index, self.p["width"],self.p["height"])
            QDS.DefObj_BoxEx(-1+self.red_index, 5000,5000)

        if self.cell_type == "on":
            QDS.SetObjColorEx([0,1,2],[self.rgb_low,self.rgb_half,self.rgb_high])
            if self.red_FF:
                QDS.SetObjColorEx([0+self.red_index,1+self.red_index,2+self.red_index],[self.rgb_low_red,self.rgb_half_red,self.rgb_high_red])
        else:
            QDS.SetObjColorEx([2,1,0],[self.rgb_low,self.rgb_half,self.rgb_high])
            if self.red_FF:
                QDS.SetObjColorEx([2+self.red_index,1+self.red_index,0+self.red_index],[self.rgb_low_red,self.rgb_half_red,self.rgb_high_red])
        QDS.SetObjColorEx([-1],[self.rgb_half])
        if self.red_FF:
            QDS.SetObjColorEx([-1+self.red_index],[self.rgb_half_red])

    def scene_render(self,duration,iobjs,opos = [(0,0)],marker_on = 0, mag = [(1,1)], angle = [0]):
        if type(iobjs) is int:
            iobjs =[iobjs]
        if not self.red_FF:
            QDS.Scene_RenderEx(duration,iobjs,opos,mag,angle*len(iobjs),marker_on)
        else:
            if marker_on:
                red_shifted = [-10+self.red_index]
                red_shifted.extend(list(np.array(iobjs)+self.red_index))
                QDS.Scene_RenderEx(duration,iobjs,[(0,0)]+opos,[(1,1)]+mag,angle*(len(iobjs)+1),marker_on)
            else:
                iobjs = [-10]+iobjs
                opos = [(0,0)]+opos
                mag = [(1,1)]+mag
                QDS.Scene_RenderEx(duration,iobjs,opos,mag,angle*len(iobjs),marker_on)
"""








