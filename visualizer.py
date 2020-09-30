# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:27:50 2019

@author: c1545871
"""
from mayavi import mlab
import numpy as np
from utils import rotate_mgrid, get_mag, get_rotational_matrix


class Visualizer:
    def __init__(self, microscope, simulation=None):
        if simulation is None:
            self.beam = None
        else:
            assert isinstance(simulation, object)
            self.beam = simulation.positions_all

        assert isinstance(microscope, object)
        self.microscope = microscope

    @staticmethod
    def create_figure():
        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5),
                    size=(480, 480))
        mlab.clf()

    def visualize_beam(self):
        x = self.beam.shape
        assert self.beam is not None
        for n in range(0, np.shape(self.beam)[0]):
            mlab.plot3d(self.beam[n, :, 0],
                        self.beam[n, :, 1],
                        self.beam[n, :, 2])

    def visualize_microscope(self):
        for element in self.microscope.elements:
            element.display()

    def visualize_field(self, field_exceptions=None, vis_type=None):
        for column in self.microscope.elements:
            for lens in column.elements:
                if lens.lens_type != field_exceptions:
                    self.field_in_lens(lens, vis_type)

    @staticmethod
    def field_in_lens(lens, vis_type=None):
        """
        Main problem here: for an angle of a column ( say, 30 deg form X-axis), I cannot make the empty grid X, Y, Z
        rotate accordingly. It just doesnt do it properly. Need to work on this one.
        Main problem #2: it seems that the y and z axes are not symmetrical when visualizing. One is accurate, the other
        is off.
        """

        if lens.R is None:
            return None
        # create grid of points
        begin = get_mag(lens.beginning)
        end = get_mag(lens.ending)

        if lens.lens_type == 'space':
            X, Y, Z = np.mgrid[begin:end:16j, -lens.R:lens.R:16j, -lens.R:lens.R:16j]
        else:
            X, Y, Z = np.mgrid[begin:end:4j, -lens.R:lens.R:4j, -lens.R:lens.R:4j]
        precision = 1e2
        X = np.round(X * precision) / precision
        Y = np.round(Y * precision) / precision
        Z = np.round(Z * precision) / precision

        # X, Y, Z = rotate_mgrid(lens.n, X, Y, Z)
        # r = np.c_[X.ravel()*np.cos(np.pi/6.) - Y.ravel()*np.sin(np.pi/6.),
        #           X.ravel()*np.sin(np.pi/6.) + Y.ravel()*np.cos(np.pi/6.), Z.ravel()]

        r = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
        # for i in range(0, 4095):
        #     r[i, :] = get_rotational_matrix(lens.n).dot(r[i, :])

        X = r[:, 0].reshape(X.shape)
        Y = r[:, 1].reshape(Y.shape)
        B = lens.getB(r)
        Bx = B[:, 0]
        By = B[:, 1]
        Bz = B[:, 2]

        Bx.shape = X.shape
        By.shape = Y.shape
        Bz.shape = Z.shape
        B_norm = np.sqrt(Bx * Bx + By * By + Bz * Bz)
        # Visualization

        # We threshold the data ourselves, as the threshold filter produce a
        # data structure inefficient with IsoSurface
        # if scale is None:
        #     B_max = 10
        # else:
        #     B_max = scale
        B_max = 4 * np.mean(B_norm)

        Bx[B_norm > B_max] = 0
        By[B_norm > B_max] = 0
        Bz[B_norm > B_max] = 0
        B_norm[B_norm > B_max] = B_max

        if vis_type == None:
            vis_type = 0

        if vis_type == 1:
            field = mlab.flow(X, Y, Z, Bx, By, Bz,
                              scalars=B_norm, name='B field')
        if vis_type == 0:
            field = mlab.pipeline.vector_field(X, Y, Z, Bx, By, Bz,
                                               scalars=B_norm, name='B field')

            vectors = mlab.pipeline.vectors(field,
                                            scale_factor=(X[1, 0, 0] - X[0, 0, 0]),
                                            )

            # Mask random points, to have a lighter visualization.
            if lens.lens_type == 'space':
                vectors.glyph.mask_input_points = True
                vectors.glyph.mask_points.on_ratio = 6

                vcp = mlab.pipeline.vector_cut_plane(field, plane_orientation="x_axes",
                                                     view_controls=True)
                vcp.glyph.glyph.scale_factor = 5 * (X[1, 0, 0] - X[0, 0, 0])

    def create(self, field=False, exceptions=None, vis_type=None):
        self.create_figure()
        self.visualize_microscope()
        self.visualize_beam()
        if field:
            self.visualize_field(field_exceptions=exceptions, vis_type=vis_type)
        mlab.show()
