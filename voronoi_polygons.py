# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:54:55 2014

Original code:
http://stackoverflow.com/questions/14144778/editing-voronoi-class-to-return-polygon-points-in-python

"""
from voronoi import voronoi
from collections import defaultdict

def voronoi_polygons(points):
    """Return series of polygons for given coordinates. """
    c = voronoi(points)
    # For each point find triangles (vertices) of a cell
    point_in_triangles = defaultdict(set)
    for t_ind, ps in enumerate(c.triangles):
        for p in ps:
            point_in_triangles[p].add(t_ind)
            
    # Vertex connectivity graph
    vertex_graph = defaultdict(set)
    for e_ind, (_, r, l) in enumerate(c.edges):
        vertex_graph[r].add(l)
        vertex_graph[l].add(r)
    
    # Caululate cells
    def cell(point):
        if point not in point_in_triangles:
            return None
        vertices = set(point_in_triangles[point]) # copy
        v_cell = [vertices.pop()]
#        vertices.add(-1)  # Simulate infinity :-)
        while vertices:
            neighbours = vertex_graph[v_cell[-1]] & vertices
            if not neighbours:
                break
            v_cell.append(neighbours.pop())
            vertices.discard(v_cell[-1])
        return v_cell
        
    # Finally, produces polygons
    polygons = []
    for i, p in enumerate(points):
        vertices = []
        point_cell = cell(i)
        for j in point_cell:
            vertices.append(c.vertices[j])
        polygons.append(tuple(vertices))
    return tuple(polygons)
    
def example():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    import misc
    x, y, z = np.loadtxt(os.path.join(misc.get_home(), 
                        "Dropbox/hydra1/spec/ulyss/kintable.txt"),
                        usecols=(1,2,5)).T
    borders = np.array([[-40., -40.], [-40., 40.], [40., -40.], [40., 40.]])
    points = np.concatenate((np.column_stack((x,y)), borders))
    z = np.concatenate((z, z.mean() * np.ones(4)))
    polygons = voronoi_polygons(points)
    coll = PolyCollection(polygons, array=z, cmap="jet", edgecolors='w') 
    background = PolyCollection(polygons[-4:], array=np.zeros(4), 
                                cmap="gray_r", edgecolors='w')
    fig, ax = plt.subplots()
    ax.add_collection(coll)
    ax.add_collection(background)
    ax.plot(x, y, "xk")
    ax.autoscale_view()
    fig.colorbar(coll, ax=ax)
    plt.xlim(35, -35)
    plt.ylim(-35, 35)
    
    return
if __name__ == "__main__":
    example()
    
    