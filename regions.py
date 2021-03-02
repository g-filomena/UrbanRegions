import networkx as nx
import pandas as pd
import numpy as np
import geopandas as gpd
import functools
import community
import array
import numbers
import warnings

from shapely.ops import polygonize_full, polygonize, unary_union
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, MultiLineString
from shapely.ops import cascaded_union, linemerge, nearest_points
pd.set_option("precision", 10)

def reset_index_dual_gdfsIG(nodesDual_gdf, edgesDual_gdf):
    '''
    The function simply reset the indexes of the two dataframes.
     
    Parameters
    ----------
    nodesDual_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
   
    Returns
    -------
    tuple of GeoDataFrames
    '''
    
    nodesDual_gdf, edgesDual_gdf = nodesDual_gdf.copy(), edgesDual_gdf.copy() 
    nodesDual_gdf = nodesDual_gdf.reset_index(drop = True)

    edgesDual_gdf['u'], edgesDual_gdf['v'] = edgesDual_gdf['u'].astype(int), edgesDual_gdf['v'].astype(int)
    nodesDual_gdf['IG_edgeID'] = nodesDual_gdf.index.values.astype(int)
    nodesDual_gdf['edgeID'] = nodesDual_gdf['edgeID'].astype(int)

    edgesDual_gdf = edgesDual_gdf.rename(columns = {'u':'old_u', 'v':'old_v'})
    edgesDual_gdf = pd.merge(edgesDual_gdf, nodesDual_gdf[['edgeID', 'IG_edgeID']], how='left', left_on='old_u', right_on='edgeID')
    edgesDual_gdf = edgesDual_gdf.rename(columns = {'IG_edgeID' : 'u'})
    edgesDual_gdf = pd.merge(edgesDual_gdf, nodesDual_gdf[['edgeID', 'IG_edgeID']], how='left', left_on='old_v', right_on='edgeID')
    edgesDual_gdf = edgesDual_gdf.rename(columns = {'IG_edgeID' : 'v'})
    edgesDual_gdf.drop(['edgeID_x', 'edgeID_y', 'old_u', 'old_v'], inplace = True, axis = 1)
    
    nodesDual_gdf.index = nodesDual_gdf['IG_edgeID']
    nodesDual_gdf.index.name = None
    
    return nodesDual_gdf, edgesDual_gdf
    
def reset_index_gdfsIG(nodes_gdf, edges_gdf):
    '''
    The function simply reset the indexes of the two dataframes.
     
    Parameters
    ----------
    nodesDual_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
   
    Returns
    -------
    tuple of GeoDataFrames
    '''
    
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()
    nodes_gdf = nodes_gdf.reset_index(drop = True)
  
    edges_gdf['u'], edges_gdf['v'] = edges_gdf['u'].astype(int), edges_gdf['v'].astype(int)
    nodes_gdf['IG_nodeID'] = nodes_gdf.index.values.astype(int)
    nodes_gdf['nodeID'] = nodes_gdf['nodeID'].astype(int)
    edges_gdf = edges_gdf.rename(columns = {'u':'old_u', 'v':'old_v'})
    edges_gdf = pd.merge(edges_gdf, nodes_gdf[['nodeID', 'IG_nodeID']], how='left', left_on='old_u', right_on=label_index)
    edges_gdf = edges_gdf.rename(columns = {'IG_nodeID' : 'u'})
    edges_gdf = pd.merge(edges_gdf, nodes_gdf[['nodeID','IG_nodeID']], how='left', left_on='old_v', right_on=label_index)
    edges_gdf = edges_gdf.rename(columns = {'IG_nodeID': 'v'})
    edges_gdf.drop(['nodeID_x', 'nodeID_y', 'old_u', 'old_v'], inplace = True, axis = 1)
    
    nodes_gdf.index = nodes_gdf['IG_nodeID']
    nodes_gdf.index.name = None
    return nodes_gdf, edges_gdf    
    

def dual_graphIG_fromGDF(nodes_dual, edges_dual):

    '''
    The function generates a NetworkX graph from dual-nodes and -edges GeoDataFrames.
            
    Parameters
    ----------
    nodes_dual: Point GeoDataFrame
        the GeoDataFrame of the dual nodes, namely the street segments' centroids
    edges_dual: LineString GeoDataFrame
        the GeoDataFrame of the dual edges, namely the links between street segments' centroids 
        
    Returns
    -------
    NetworkX Graph
    '''
   
    edges_dual.u = edges_dual.u.astype(int)
    edges_dual.v = edges_dual.v.astype(int)
    
    Dg = nx.Graph()   
    Dg.add_nodes_from(nodes_dual.index)
    attributes = nodes_dual.to_dict()
    
    a = (nodes_dual.applymap(type) == list).sum()
    if len(a[a>0]): 
        to_ignore = a[a>0].index[0]
    else: to_ignore = []
    
    for attribute_name in nodes_dual.columns:
        # only add this attribute to nodes which have a non-null value for it
        if attribute_name in to_ignore: 
            continue
        attribute_values = {k:v for k, v in attributes[attribute_name].items() if pd.notnull(v)}
        nx.set_node_attributes(Dg, name=attribute_name, values=attribute_values)

    # add the edges and attributes that are not u, v, key (as they're added
    # separately) or null
    for _, row in edges_dual.iterrows():
        attrs = {}
        for label, value in row.iteritems():
            if (label not in ['u', 'v']) and (isinstance(value, list) or pd.notnull(value)):
                attrs[label] = value
        Dg.add_edge(row['u'], row['v'], **attrs)

    return Dg
    ## polygonise
    
def weight_nodes_gdf(nodes_gdf, service_points_gdf, list_amenities, field, radius = 50):
    """
    Given a nodes- and a services/points-geodataframes, the function assigns an attribute to nodes in the graph G (prevously derived from 
    nodes_gdf) based indeed on the amount of features in the services_gdf in a buffer around each node. 
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
    service_points_gdf: Point GeoDataFrame
    G: networkx multigraph
    name: string, attribute name
    radius: float, distance around the node within looking for point features (services)
    
    Returns
    -------
    networkx multidigraph
    """
    nodes_gdf[field] = None
    tmp = service_points_gdf[service_points_gdf['amenity']].copy()
    sindex = tmp.sindex
    
    nodes_gdf[field] = nodes_gdf.apply(lambda row: _count_services_around_node(row["geometry"], tmp, sindex, radius = radius), axis=1)
    
    return nodes_gdf

def _count_services_around_node(node_geometry, service_points_gdf, service_points_gdf_sindex, radius):
    """
    The functions supports the weight_nodes function.
    
    Parameters
    ----------
    node_geometry: Point geometry
    service_points_gdf: Point GeoDataFrame
    service_points_gdf_sindex = Rtree Spatial Index
    radius: float, distance around the node within looking for point features (services)
    
    Returns
    -------
    Integer value
    """

    buffer = node_geometry.buffer(radius)
    possible_matches_index = list(service_points_gdf_sindex.intersection(buffer.bounds))
    possible_matches = service_points_gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(buffer)]
    weight = len(precise_matches)
        
    return weight
    
def polygonise_clusters(points_gdf, partition_field, crs):
    polygons = []
    partitionIDs = []
    d = {'geometry' : polygons, 'partitionID' : partitionIDs}

    partitions = points_gdf[partition_field].unique()
    for i in partitions:
        polygon =  points_gdf[points_gdf[partition_field] == i].geometry.unary_union.convex_hull
        polygons.append(polygon)
        partitionIDs.append(i)

    df = pd.DataFrame(d)
    partitions_polygonised = gpd.GeoDataFrame(df, crs=crs, geometry=df['geometry'])
    return partitions_polygonised