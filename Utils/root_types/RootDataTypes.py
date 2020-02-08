#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jhorn
"""

from xml.etree import ElementTree as xml_et
import os
import parser
from math import *

home = os.path.expanduser( "~" )

def from3DEntryToList( elem, dt = float ):
    return [dt(elem.get("x")), dt(elem.get("y")), dt(elem.get("z"))]

def from3DEntryToString( elem ):
    return "{0}x{1}x{2}".format( elem.get("x"), elem.get("y"), elem.get("z") )

def fromEntryToBool( elem ):
    val = elem.text
    if val == "0": return False
    else: return True

def parseMathList( list_elem, ndigits=None ):
    list_str = list_elem.text.replace( " ", "" )
    eq_str_list = list_str.split(",")
    val_list = []
    for eq_str in eq_str_list:
        eq = parser.expr( eq_str ).compile()
        res = eval( eq )
        if ndigits is not None:
            res = round( res, ndigits=ndigits )
        val_list.append( res )
    return val_list

def getAllDGenInfos():
    tags = _root_data_types.getAll()
    infos = []
    for tag in tags:
        infos.append( DataGenerationInfo( tag ) )
    return infos


class DataGenerationInfo:
    def __init__( self, dt_string ):
        full_data = _root_data_types( dt_string )
        self.data_name = dt_string
        self.dir_name = full_data.find( "dir_name" ).text
        self.size_name = from3DEntryToString( full_data.find( "shape" ) )
        self.shape = from3DEntryToList( full_data.find( "shape" ), int )
        self.radius_multiplier = parseMathList( full_data.find( "radius_multiplier" ), 2 )
        mri_xml = full_data.find( "real_mri" )
        self.mri_path = mri_xml.find( "path" ).text
        self.mri_name = mri_xml.find( "name" ).text
        self.depth_axis = full_data.find( "depth_axis" ).text
        self.swap_x_z = fromEntryToBool( full_data.find( "swap_x_z" ) )
        mask_xml = full_data.find( "mask" )
        pot_xml = mask_xml.find( "plant_pot" )
        self.pot_pos = [float( pot_xml.get( "x" ) ), float( pot_xml.get( "y" ) )]
        self.pot_radius = float( pot_xml.get( "radius" ) )
        tube_xml = mask_xml.find( "test_tube" )
        self.tube_pos = [float( tube_xml.get( "x" ) ), float( tube_xml.get( "y" ) )]
        self.tube_rad = [float( tube_xml.get( "outer_radius" ) ), float( tube_xml.get( "inner_radius" ) )]
        
        
class VoxelizationInfo:
    def __init__( self, dt_string ):
        full_data = _root_data_types( dt_string )
        self.data_name = dt_string
        self.dir_name = full_data.find( "dir_name" ).text
        self.shape = from3DEntryToList( full_data.find( "shape" ), int )
        self.radius_multiplier = parseMathList( full_data.find( "radius_multiplier" ) )
        vox_xml = full_data.find( "voxelization" )
        self.xml_path = home +"/" +vox_xml.find( "xml_path" ).text
        self.translation = from3DEntryToList( vox_xml.find( "translation" ) )
        self.min = from3DEntryToList( vox_xml.find( "min" ) )
        self.max = from3DEntryToList( vox_xml.find( "max" ) )
        self.axis_multiplier = parseMathList( vox_xml.find( "axis_multiplier" ) )
        self.z_fact = float( vox_xml.find( "z_fact" ).text )
        self.fact = float( vox_xml.find( "fact" ).text )
    
    
class RootDataTypes:
    
    def __init__( self, path="" ):
        data_xml = xml_et.parse( path +"/RootDataTypes.xml" )
        self.root = data_xml.getroot()
        self.type_list = []
        for child in self.root:
            tag = child.tag
            if tag[0] == "_":
                tag = tag[1:]
            self.type_list.append( tag )
        
    def getDataType( self, dt_string ):
        if dt_string[0].isdigit():
            dt_st = "_" +dt_string
        else:
            dt_st = dt_string 
        for child in self.root:
            if dt_st == child.tag:
                if child.find( "parent_type" ) is not None:
                    return self.fromParentType( child )
                # print( "Found datatype \"{0}\" in xml.".format( dt_string ) )
                return child
        raise RuntimeError( "Could not find \"{0}\". DataTypes available are: {1}".format( dt_string, self.type_list ) )

    def fromParentType( self, child_type ):
        data = self.getDataType( child_type.find( "parent_type" ).text )
        data.find( "dir_name" ).text = child_type.find( "dir_name" ).text
        mri = data.find( "real_mri" )
        mri.find( "path" ).text = child_type.find( "mri_path" ).text
        vox = data.find( "voxelization" )
        vox.find( "xml_path" ).text = child_type.find( "xml_path" ).text
        return data
    
    def __call__( self, dt_string ):
        return self.getDataType( dt_string )

    def getAll( self ):
        tag_list = []
        for child in self.root:
            if child.tag[0] == '_':
                tag_list.append( child.tag[1:] )
            else:
                tag_list.append( child.tag )
        return tag_list       

_root_data_types = RootDataTypes( os.path.dirname( __file__ ) )                
if __name__ == "__main__":
    dt_type = "ii_sand_d4_dap5"
    dt = VoxelizationInfo( dt_type )
    print( dt.xml_path )
    print( dt.shape )
    print( dt.radius_multiplier )
    print( dt.translation )
    print( dt.min )
    print( dt.max )
    print( dt.z_fact )
    print( dt.fact )
    dt = DataGenerationInfo( dt_type )
    print( dt.radius_multiplier )
    print( dt.dir_name )
    print( dt.mri_path )
    print( dt.mri_name )
    print( dt.depth_axis )
    print( dt.swap_x_z )
    print( dt.pot_pos )
    print( dt.pot_radius )
    print( dt.tube_pos )
    print( dt.tube_rad )
    
