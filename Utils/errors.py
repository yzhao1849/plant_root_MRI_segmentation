class Errors(Exception):
    """ A base class for all custom errors """
    pass


class CropSizeInvalidError(Errors):
    """ When the user specified crop side length is not in [10,20,40,80]"""
    pass


class ArtifactUndefinedError(Errors):
    """ Raise when the direction of the artifact to be added to the input image 
    is not in ['xy_plane', 'yz_plane', 'xz_plane']"""
    pass


class InvalidTypeError(Errors):
    """ Raise when the direction of the artifact to be added to the input image 
    is not in ['xy_plane', 'yz_plane', 'xz_plane']"""
    pass

