# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:15:23 2016

A simple GeoHash implementation.

@author: roya
"""


# Forward and reverse base 32 map
baseseq = '0123456789bcdefghjkmnpqrstuvwxyz'
base32map = dict((k,count) for count,k in enumerate(baseseq))
base32map_reverse = dict((count,k) for count,k in enumerate(baseseq))

def _bits_to_float(bits, lower=-90.0, middle=0.0, upper=90.0):
  """Convert GeoHash bits to a float."""
  for i in bits:
    if i:
      lower = middle
    else:
      upper = middle
    middle = (upper + lower) / 2
  return middle
  
def _float_to_bits(value, lower=-90.0, middle=0.0, upper=90.0, length=15):
  """Convert a float to a list of GeoHash bits."""
  bits = []
  for i in range(length):
    if value >= middle:
      lower = middle
      bits.append(1)
    else:
      upper = middle
      bits.append(0)
    middle = (upper + lower) / 2
  return bits

def _geohash_to_bits(value):
  """Convert a GeoHash to a list of GeoHash bits."""
  b = map(base32map.get, value)
  bits = []
  for i in b:
    out = []
    for z in range(5):
      out.append(i & 0b1)
      i = i >> 1
    bits += out[::-1]
  return bits
  
def _bits_to_geohash(value):
  """Convert a list of GeoHash bits to a GeoHash."""
  strhash = []
  # Get 5 bits at a time
  for i in (value[i:i+5] for i in xrange(0, len(value), 5)):
    # Convert binary to integer
    # Note: reverse here, the slice above doesn't work quite right in reverse.
    total = sum([(bit*2**count) for count,bit in enumerate(i[::-1])])
    strhash.append(base32map_reverse[total])
  # Join the string and return
  return "".join(strhash)

# Public      
def decode(value):
  """Decode a geohash. Returns a (lon,lat) pair."""
  assert value, "Invalid geohash: %s"%value
  # Get the GeoHash bits
  bits = _geohash_to_bits(value)
  # Unzip the GeoHash bits.
  lon = bits[0::2]
  lat = bits[1::2]
  # Convert to lat/lon
  return (
    _bits_to_float(lon, lower=-180.0, upper=180.0),
    _bits_to_float(lat)
  )

def encode(lonlat, length=12):
  """Encode a (lon,lat) pair to a GeoHash."""
  assert len(lonlat) == 2, "Invalid lon/lat: %s"%lonlat
  # Half the length for each component.
  length /= 2
  lon = _float_to_bits(lonlat[0], lower=-180.0, upper=180.0, length=length*5)
  lat = _float_to_bits(lonlat[1], lower=-90.0, upper=90.0, length=length*5)
  # Zip the GeoHash bits.
  ret = []
  for a,b in zip(lon,lat):
    ret.append(a)
    ret.append(b)
  return _bits_to_geohash(ret)

def adjacent(geohash, direction):
  """Return the adjacent geohash for a given direction."""
  # Based on an MIT licensed implementation by Chris Veness from:
  #   http://www.movable-type.co.uk/scripts/geohash.html
  assert direction in 'nsew', "Invalid direction: %s"%direction
  assert geohash, "Invalid geohash: %s"%geohash
  neighbor = {
    'north': [ 'p0r21436x8zb9dcf5h7kjnmqesgutwvy', 'bc01fg45238967deuvhjyznpkmstqrwx' ],
    'south': [ '14365h7k9dcfesgujnmqp0r2twvyx8zb', '238967debc01fg45kmstqrwxuvhjyznp' ],
    'east': [ 'bc01fg45238967deuvhjyznpkmstqrwx', 'p0r21436x8zb9dcf5h7kjnmqesgutwvy' ],
    'west': [ '238967debc01fg45kmstqrwxuvhjyznp', '14365h7k9dcfesgujnmqp0r2twvyx8zb' ]
  }
  border = {
    'north': [ 'prxz',     'bcfguvyz' ],
    'south': [ '028b',     '0145hjnp' ],
    'east': [ 'bcfguvyz', 'prxz'     ],
    'west': [ '0145hjnp', '028b'     ]
  }
  last = geohash[-1]
  parent = geohash[0:-1]
  t = len(geohash) % 2
  # Check for edge cases
  if (last in border[direction][t]) and (parent):
    parent = adjacent(parent, direction)
  return parent + baseseq[neighbor[direction][t].index(last)]

def neighbors(geohash):
  """Return all neighboring geohashes."""
  return {
    'n':  adjacent(geohash, 'north'),
    'ne': adjacent(adjacent(geohash, 'north'), 'east'),
    'e':  adjacent(geohash, 'east'),
    'se': adjacent(adjacent(geohash, 'south'), 'east'),
    's':  adjacent(geohash, 'south'),
    'sw': adjacent(adjacent(geohash, 'south'), 'west'),
    'w':  adjacent(geohash, 'west'),
    'nw': adjacent(adjacent(geohash, 'north'), 'west'),
    'c':  geohash
  }

def neighborsfit(centroid, points):
  centroid = encode(centroid)
  points = map(encode, points)
  for i in range(1, len(centroid)):
    g = centroid[0:i]
    n = set(neighbors(g).values())
    unbounded = [point for point in points if (point[0:i] not in n)]
    if unbounded:        
      break
  return g[0:-1]