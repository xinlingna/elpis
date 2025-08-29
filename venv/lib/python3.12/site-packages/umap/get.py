# -*- coding: utf-8 -*-
from vincenty import vincenty
from bson.son import SON

from IPython.display import HTML
import folium


def distance(source, target):
    """
    1 = 1km
    :param source:
    :param target:
    :return:
    """
    return vincenty(source, target)

def nearest(database, name, location, maxkm):
    """
    MongoDB GeoSpatial Search
    :param database: db.test
    :param name: field name 'gps'
    :param location: [lat, lon]
    "All documents must store location data in the same order.
    If you use latitude and longitude as your coordinate system,
    always store longitude first.
    MongoDB’s 2d spherical index operators only recognize
    [longitude, latitude] ordering."
    :param maxkm: limit of distance
    :return: count, database
    """
    query = {name: SON([("$near", location),
                         ("$maxDistance", maxkm/111.12)])}
    data = database.find(query)
    return data.count(), data

# def nearest(data, location, dist=10):
#     for d in data:
#         cand = vincenty(location, d['loc'])
# #         print dist, cand
#         if dist > cand:
#             dist = cand
# #             result = d
#     return dist



# 시장_쇼핑센터
# one degree is approximately 111.12 kilometers
# for m in MB:
#     maps = db.maps.find({'cat': m})
#     for mm in maps:
#         location = mm['loc']
#         query = {"loc": SON([("$near", location), ("$maxDistance", 0.001/111.12)]), "cat": "시장_쇼핑센터"}
#         data = db.maps.find(query)
#         count = data.count()
#         ite = 1
#         while count < 1:
#             query = {"loc": SON([("$near", location), ("$maxDistance", (0.001+0.1*ite)/111.12)]), "cat": "시장_쇼핑센터"}
#             data = db.maps.find(query)
#             count = data.count()
#             ite += 1
# #         print data.count(),
#         dist = nearest(data, location)
#         db.maps.update({'_id':mm['_id']}, {'$set': {'market': dist}})
#     print m





def inline_map(map):
    """
    Embeds the HTML source of the map directly into the IPython notebook.

    This method will not work if the map depends on any files (json data). Also this uses
    the HTML5 srcdoc attribute, which may not be supported in all browsers.
    """
    if isinstance(map, folium.Map):
        map._build_map()
        srcdoc = map.HTML.replace('"', '&quot;')
        embed = HTML('<iframe srcdoc="{srcdoc}" '
                     'style="width: 100%; height: 500px; '
                     'border: none"></iframe>'.format(srcdoc=srcdoc))
    else:
        raise ValueError('{!r} is not a folium Map instance.')
    return embed

def embed_map(map, path="map.html"):
    path = path
    """
    Embeds a linked iframe to the map into the IPython notebook.

    Note: this method will not capture the source of the map into the notebook.
    This method should work for all maps (as long as they use relative urls).
    """
    map.create_map(path=path)
    return HTML('<iframe src="files/{path}" style="width: 100%; height: 510px; border: none"></iframe>'.format(path=path))

# def export_map(map, path="export_data.html", js="test.js"):
#     path = path
#     map.create_map(path=path)
#
#     # Open the HTML file
#     f = open(path, 'r')
#     text = f.read()
#     f.close()
#
#     # Parsing HTML for circles
#     pos = text.find('var circle_1')
#     if len(splited) > 1:
#         circles = text[pos:]
#     else:
#         print 'Length is 1'
#     circles = circles.replace('</script>','').replace('</body>','')
#     cv = ''
#     vs = []
#     for i in circles.split('\n'):
#         if 'map' not in i:
#             cv += i.replace('  ','')+'\n'
#         if 'var ' in i:
#             i = i.replace('  ','')
#             vs.append(i.split(' ')[1])
#     # Making a new JS file
#     head = """var circle_group = L.layerGroup("""
#     vs = str(vs).replace("'", '')
#     tail = """);"""
#     java_1 = cv
#     java_2 = head+vs+tail
#     f = open(js, 'w')
#     filename = js.split('.')[0]+'_'
#     f.write(java_1.replace('circle_', filename))
#     f.write(java_2.replace('circle_', filename))
#     f.close()
#     return js


class Map():
    """
    Map with folium
    """
    def __init__(self):
        """
        lat, lon, zoom_start
        """
        self.lat = 37.27
        self.lon = 127.01
        self.zoom_start = 7
        self.m = folium.Map(location=[self.lat, self.lon], zoom_start=self.zoom_start)
        self.circles = []

    def reset(self):
        """
        reset circles, map
        :return:
        """
        self.circles = []
        self.m = folium.Map(location=[self.lat, self.lon], zoom_start=self.zoom_start)

    def drawing(self):
        """
        Map drawing
        :return: iframe map
        """
        if len(self.circles)>0:
            for c in self.circles:
                self.m.circle_marker(location=c[0], radius=c[1], fill_opacity=c[2],
                                     popup=c[3], fill_color=c[4], line_color=c[5])
        self.m._build_map()
        srcdoc = self.m.HTML.replace('"', '&quot;')
        embed = HTML('<iframe srcdoc="{srcdoc}" '
                     'style="width: 100%; height: 500px; '
                     'border: none"></iframe>'.format(srcdoc=srcdoc))
        return embed

    def add_circle(self, location, radius=50, fill_opacity=0.8,
                   popup='', fill_color='black', line_color='None'):
        """
        Add a circle in the map
        :param location: [lat, lon]
        :param radius: circle radius
        :param fill_opacity: transparency
        :param popup: text
        :param fill_color:
        :param line_color: circle line color
        :return:
        """
        if type(popup) == unicode:
            popup = popup.encode('utf8')
        self.circles.append([location[::-1], radius, fill_opacity, popup, fill_color, line_color])

    def get_circles(self):
        """
        Get a list of circles
        :return: list
        """
        return self.circles
