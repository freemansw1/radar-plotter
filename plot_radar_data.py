from asyncio import Task
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pyart
import numpy as np
from datetime import timedelta
from datetime import datetime
import datetime
import pandas as pd
import netCDF4 as nc
import glob
import cartopy.crs as ccrs
import cartopy.feature as feat

import cartopy
from pyart.graph import cm
import pytz
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

from metpy.calc import wind_components
from metpy.plots import StationPlot, StationPlotLayout, simple_layout
from metpy.units import units
from netCDF4 import Dataset
from matplotlib.colors import LinearSegmentedColormap

# for interfacing with AWS
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
from jug import TaskGenerator
import jug

# User options

radarname = 'KCYS'
# Where do you want to download the radar data to?
# If this directory doesn't exist, the script will create it. 
download_dir='./'+radarname+"_data/"
# Where do you want the plots to output to?
plot_dir = './'+radarname+"_plots_vel/"
# what prefix to add to the output file names
plot_prefix = radarname+'_vel05'
# Where are the radiosonde files?
sonde_dir = './sonde_data/'
# Metar files from https://mesonet.agron.iastate.edu/request/download.phtml?
# note that you must download with lat/lon. 
# also I suggest downloading 5-minute data. 
metar_dir = './metar_data/'

# topography file (using ETOPO1 from: https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/grid_registered/netcdf/)
topography_file = './ETOPO1_Ice_g_gmt4.grd'

# county shapefile
# https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html
# using the 5m (1:5,000,000) here
county_shapefile = './cb_2021_us_county_5m/cb_2021_us_county_5m.shp'

# Download roads shapefiles from: https://www.census.gov/cgi-bin/geo/shapefiles/index.php
# primary and secondary roads by state

# CO roads shapefile
co_roads_shapefile = './tl_2021_08_prisecroads/tl_2021_08_prisecroads.shp'
# WY roads shapefile
wy_roads_shapefile = './tl_2021_56_prisecroads/tl_2021_56_prisecroads.shp'
# NE roads shapefile
ne_roads_shapefile = './tl_2021_31_prisecroads/tl_2021_31_prisecroads.shp'

all_road_shapefiles = [co_roads_shapefile, wy_roads_shapefile, ne_roads_shapefile]

# datetimes in UTC
start_datetime = datetime.datetime(2022, 6, 6, 18)
end_datetime = datetime.datetime(2022, 6, 6, 19)

# plotting options
sonde_size=90
sndcolors=[ 'cyan', 'magenta', 'k']

# Latitude/Longitude boundaries
lonmin = -106
lonmax = -102
latmin = 39.5
latmax = 42.5

# maximum time away from plot time that the metar station plot can be
max_station_time = timedelta(hours=1)
#max_station_time = timedelta(minutes=5)

# What you want the first line of the title to be
title_firstpart = r""+radarname+r" Reflectivity 0.5$^\circ$ and sonde locations"

# TODO: Include option for other variables

# options: 'reflectivity', 'velocity'
var_to_plot = 'reflectivity'

# minimum and maximum values to plot
# make sure that these match the color table
vmin, vmax = -30, 95
#for velocity
#vmin, vmax = -120, 120

# for base reflectivity, this is usually 0, and for 
# base velocity, this is usually 1. 
sweep_number = 0

# reflectivity color table
grctable_ref = """color: -30 116 78 173 147 141 117
color: -20 150 145 83 210 212 180
color: -10 204 207 180 65 91 158
color: 10 67 97 162 106 208 228
color: 18 111 214 232 53 213 91
color: 22 17 213 24 9 94 9
color: 35 29 104 9 234 210 4 
color: 40 255 226 0 255 128 0
color: 50 255 0 0 113 0 0
color: 60 255 255 255 255 146 255
color: 65 255 117 255 225 11 227
color: 70 178 0 255 99 0 214
color: 75 5 236 240 1 32 32
color: 85 1 32 32
color: 95 1 32 32"""

# Velocity color table
grctable_vel = """color: -120 252 0 130 109 2 150 
color: -100 110 3 151 22 13 156
color: -90  24 39 165 30 111 188
color: -80 30 111 188 40 204 220
color: -70 47 222 226 181 237 239 
color: -50 181 237 239 2 241 3
color: -40 3 234 2  0 100 0
color: -10 78 121 76 116 131 112
color: 0 137 117 122 130 51 59 
color: 10 109 0 0 242 0 7
color: 40 249 51 76 255 149 207
color: 55 253 160 201 255 232 172
color: 60 253 228 160 253 149 83 
color: 80 254 142 80 110 14 9
color: 120 110 14 9
"""


grctable = grctable_ref 

# TODO: include a star where CPER is. 
# location of main location (e.g., CPER) in (latitude, longitude) format
# Set this to None to not plot it. 
main_loc = (40.80985556259899, -104.7782733197491)
# options for plotting the main location point
main_loc_plot_opts = {
    'marker': '*',
    'facecolor': 'k',
    'edgecolor': 'k',
    's': 40
}


def daterange(start_date, end_date):
    for n in range(int(((end_date.date()) - (start_date.date())).days+1)):
        yield start_date.date() + datetime.timedelta(n)


def parse_sonde(sonde_folder):
    sonde_file = glob.glob(sonde_folder+'/*.dat')
    sonde = pd.read_csv(open(sonde_file[0],
                                errors='ignore'),
                          parse_dates=['Date+Time'], index_col=False)
    sonde = sonde.drop_duplicates(subset=['Sample#'], keep='last')
    return sonde

def get_sonde_locs(radar_time, sondelist):
    sondelocs = list()
    for sonde in sondelist:
        sndname = sonde[1].split('/')[-1]
        sonde = sonde[0]
        dt= sonde['Date+Time']-radar_time

        if max(abs(dt)<timedelta(minutes=2)) == True:
            print(sndname)


            #we can plot this sonde.
            i = np.argmin(np.abs(sonde['Date+Time'] - radar_time))
            sonde_loc = sonde.iloc[i]
            #print(sonde_loc)
            valid = False
            ntries = 0
            maxtries = 120
            offsettry = ntries
            while not valid and ntries < maxtries and (i+offsettry+1):
                if ntries> maxtries//2:
                    offsettry = -1* (ntries%(maxtries//2))
                else:
                    offsettry = ntries
                try:
                    sonde_loc = sonde.iloc[i+offsettry]
                except IndexError:
                    print("beyond index")
                ntries = ntries + 1
                #print(ntries)
                
                if sonde_loc['Lat']>1000 or sonde_loc['Long']>1000:

                    #print("error sonde", sndname, i)
                    pass
                else:
                    valid = True
                    sondelocs.append({'Lat':sonde_loc['Lat'], 'Lon':sonde_loc['Long'],
                             'Alt':sonde_loc['Alt'],'Name':sndname })
            if not valid:
                print("error sonde final", sndname, i, ntries )
                continue



        else:
            #this sonde is gone. 
            continue
            
    return sondelocs


from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    try:
        plt.cm.get_cmap(name)
    except ValueError:
        plt.register_cmap(cmap=newcmap)

    return newcmap


def convert_gr_table(grstr):
    '''
    Convert a color table designed for GRLevel2/3 to a python one.
    Be sure that the min/max values are identical. 
    '''
    spstr = grstr.split("color:")
    spstr = [x.strip() for x in spstr]
    varvalues = list()
    red1values = list()
    red2values = list()
    blue1values = list()
    blue2values = list()
    green1values = list()
    green2values = list()

    for interval in spstr:
        if interval == '':
            continue
        indivals = interval.split()
        varvalues.append(int(indivals[0]))
        red1values.append(int(indivals[1]))
        green1values.append(int(indivals[2]))
        blue1values.append(int(indivals[3]))
        if len(indivals)<5:
            #we aren't discontinuous here.
            red2values.append(-1)
            green2values.append(-1)
            blue2values.append(-1)
        else:
            red2values.append(int(indivals[4]))
            green2values.append(int(indivals[5]))
            blue2values.append(int(indivals[6]))
    
    normvarvals = [(x+(0-min(varvalues)))/
                   (max(varvalues)+(0-min(varvalues))) for x in varvalues]
    red1values = [x/255.0 for x in red1values]
    red2values = [x/255.0 for x in red2values]
    green1values = [x/255.0 for x in green1values]
    green2values = [x/255.0 for x in green2values]
    blue1values = [x/255.0 for x in blue1values]
    blue2values = [x/255.0 for x in blue2values]
    redvals = list()
    greenvals = list()
    bluevals = list()
    for i, num in enumerate(normvarvals):
        if i == 0:
            redvals.append((num, 0.0, red1values[i]))
            greenvals.append((num, 0.0, green1values[i]))
            bluevals.append((num, 0.0, blue1values[i]))
        
        else:
            if red2values[i-1]<0:
                redvals.append((num, red1values[i], red1values[i]))
                greenvals.append((num, green1values[i], green1values[i]))
                bluevals.append((num, blue1values[i], blue1values[i]))

            else:
                redvals.append((num, red2values[i-1], red1values[i]))
                greenvals.append((num, green2values[i-1], green1values[i]))
                bluevals.append((num, blue2values[i-1], blue1values[i]))

    cmapdict = {
        'red':tuple(redvals),
        'green':tuple(greenvals),
        'blue':tuple(bluevals)
    }
    return cmapdict

@TaskGenerator
def plot_radar_data(radar_file, metar_data, sondes, terrain_data):

    filename=radar_file
    print(filename)

    # set some plotting info
    import matplotlib 
    matplotlib.rc('xtick', labelsize=16) 
    matplotlib.rc('ytick', labelsize=16) 
    matplotlib.rc('font', size=16) 

    wdtbtable = convert_gr_table(grctable)
    try:
        plt.cm.get_cmap('wdtbtable')
    except ValueError:
        wdbt = LinearSegmentedColormap('wdtbtable', wdtbtable)
        plt.register_cmap(cmap=wdbt)


    fig = plt.figure(figsize=(15, 10))
    proj = ccrs.LambertConformal(central_longitude=-104, central_latitude=40
                                    ,standard_parallels=[35])
    #proj = ccrs.PlateCarree()


    ax = fig.add_subplot(1, 1, 1, projection=proj)

    ax.set_extent((lonmin, lonmax, latmin, latmax))


    # Add some various map elements to the plot to make it recognizable
    radar = pyart.io.read_nexrad_archive(filename)
    display = pyart.graph.RadarMapDisplay(radar)


    display.plot_ppi_map(var_to_plot, sweep=sweep_number, vmin=vmin, vmax=vmax, ax=ax,         
                            mask_outside=True, cmap = 'wdtbtable', ticks=np.arange(vmin,vmax+1,10))


    ax = display.ax



    # Get relevant shapefiles
    counties = cartopy.io.shapereader.Reader(county_shapefile)
    ax.add_geometries(counties.geometries(), ccrs.PlateCarree(),
                        edgecolor='grey', facecolor='None', linewidth=0.5)
    for road_shapefile in all_road_shapefiles:
        roads = cartopy.io.shapereader.Reader(road_shapefile)
        ax.add_geometries(roads.geometries(), ccrs.PlateCarree(),
                        edgecolor='blue', facecolor='None', linewidth=0.3)

    radar_time = nc.num2date(radar.time['data'][0], radar.time['units'],
                            only_use_cftime_datetimes=False,
                            only_use_python_datetimes=True)
    shrunk_cmap = shiftedColorMap(matplotlib.cm.terrain, start=0.3, midpoint=0.5, stop=0.95, name='shrunk_terrain')

    tercont = ax.contourf(terrain_data['x'], terrain_data['y'], terrain_data['z'], 
                levels=np.linspace(1000,2750,70), cmap='shrunk_terrain', zorder=0, extend='both',
                transform_first=True)



    sondesinair = get_sonde_locs(radar_time, sondes)

    gb = metar_data.groupby('station')
    # find the closest valid observation time for each station
    def f(x):
        return pd.Series([np.abs(x-radar_time)], index = ['difftime'])
    sfcstations = gb['valid'].apply(f)

    stationsdf = pd.DataFrame()
    all_station_rows = list()
    for station in sfcstations:
        closest = station.idxmin()
        closestob = metar_data.loc[closest]
        
        if np.abs(closestob['valid'] - radar_time)<max_station_time:
            #within an hour.
            all_station_rows.append(closestob)
    
    stationsdf = pd.DataFrame(all_station_rows)
    stationsdf = stationsdf.replace('M', np.nan)
    datamsk = ((stationsdf['lat']>latmin+0.1)& (stationsdf['lat']<latmax) & 
                (stationsdf['lon']>lonmin)& (stationsdf['lon']<lonmax))
    data = stationsdf[datamsk]
    # Start the station plot by specifying the axes to draw on, as well as the
    # lon/lat of the stations (with transform). We also the fontsize to 12 pt.

    stationplot = StationPlot(ax, data['lon'].values, data['lat'].values, transform=ccrs.PlateCarree(),
                                fontsize=11, spacing=11)

    # Plot the temperature and dew point to the upper and lower left, respectively, of
    # the center point. Each one uses a different color.
    stationplot.plot_parameter('NW', data['tmpf'].values.astype(float), color='red')
    stationplot.plot_parameter('SW', data['dwpf'].values.astype(float), color='darkgreen')


    u, v = wind_components((data['sknt'].values.astype(float) * units('knots')),
                                data['drct'].values.astype(float) * units.degree)

    # Add wind barbs
    stationplot.plot_barb(np.array(u), np.array(v))

    #FOR CAMPUS WX STATION ONLY 40.576238, -105.085729
    # uncomment to plot campus weather station
    '''

    def f(x):
        return pd.Series([np.abs(x-radar_time)], index = ['difftime'])
    campusstn = campuswxstn['UTC_date'].apply(f)

    closest = campusstn['difftime'].argmin()
    campusob = campuswxstn.ix[closest]


    camstationplot = StationPlot(ax, np.array([-105.085729]), np.array([40.576238]), transform=ccrs.PlateCarree(),
                                fontsize=11, spacing=11)
    

    # Plot the temperature and dew point to the upper and lower left, respectively, of
    # the center point. Each one uses a different color.
    camstationplot.plot_parameter('NW', [campusob['Temp']], color='red')
    camstationplot.plot_parameter('SW', [campusob['DewPt']], color='darkgreen')


    u, v = wind_components((np.array(campusob['Wind'])*0.868976242 * units('knots')),
                                np.array(campusob['Dir']) * units.degree)

    # Add wind barbs
    camstationplot.plot_barb(np.array(u), np.array(v))

    '''

    # Plot primary site
    ax.scatter(main_loc[1], main_loc[0], transform=ccrs.PlateCarree(), **main_loc_plot_opts)

    legplots = list()
    legnames = list()

    for airsonde, sndcolor in zip(sondesinair, sndcolors):
        sndsca = ax.scatter(airsonde['Lon'],airsonde['Lat']
                            ,marker='o', facecolor='cyan', edgecolor='k', s=sonde_size, transform=ccrs.PlateCarree())

        legplots.append(sndsca)
        legnames.append(airsonde['Name']+" @ "+str(round(airsonde['Alt']))+"m MSL")

    mountain_tz = pytz.timezone('America/Denver')
    curr_mountain_time = radar_time.replace(tzinfo=datetime.timezone.utc).astimezone(tz=mountain_tz)
    plt.title(title_firstpart+"\n"+
                r"at "+radar_time.strftime("%m/%d/%y %H:%M:%S Z")+", "+curr_mountain_time.strftime("%H:%M:%S MT"), size=20)


    cbaxes = fig.add_axes([0.2, 0.05, 0.6, 0.02])  # This is the position for the colorbar
    cb = plt.colorbar(tercont, cax = cbaxes, ticks=np.arange(1000,2751,250), orientation='horizontal')
    #cbaxes.yaxis.set_ticks(np.arange(1000,2500,250))
    #cbaxes.yaxis.set_ticks_position('left')
    cb.set_label('Elevation (m)')


    
    leg = ax.legend(legplots,
                legnames,
                scatterpoints=1,
                loc='upper center',
                ncol=3,
                fontsize=14,
                bbox_to_anchor=(0.5, -0.14),
                fancybox=True, shadow=True)

    


    display.plot_point(radar.longitude['data'][0], radar.latitude['data'][0])


    plt.savefig(plot_dir+
        radarname+"_ref05_"+radar_time.strftime("%y%m%d_%H%M%S")+
        ".png", dpi=160, bbox_inches="tight")

    plt.close(fig)

@TaskGenerator
def get_radar_data(radarname, start_datetime, end_datetime, download_dir):
    '''gets the radar data from an AWS bucket
    '''
    s3 = boto3.resource('s3', region_name='us-east-1', config=Config(signature_version=UNSIGNED))
    nexrad_bucket = s3.Bucket('noaa-nexrad-level2')

    # get all radar bucket objects
    all_radar_data = list()
    for curr_date in daterange(start_datetime, end_datetime):
        #print(curr_date)
        curr_prefix = curr_date.strftime("%Y/%m/%d/")+radarname
        curr_radar_data = list(nexrad_bucket.objects.filter(Prefix=curr_prefix))
        radar_dates = [datetime.datetime.strptime(in_obj.key.split('/')[-1].split('V06')[0], radarname+"%Y%m%d_%H%M%S_") for in_obj in curr_radar_data]
        dates_in_range_sel = np.logical_and(np.array(radar_dates)>start_datetime, np.array(radar_dates)<end_datetime)
        all_radar_data+=np.array(curr_radar_data)[dates_in_range_sel].tolist()

    # Download radar data from AWS
    radfiles = list()
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)
    for i, curr_rad_obj in enumerate(all_radar_data):
        radar_out_filename = download_dir+curr_rad_obj.key.split('/')[-1]
        if 'MDM' in radar_out_filename:
            continue

        # if we have already downloaded it, don't download it again. 
        if os.path.exists(radar_out_filename):
            pass
        else:
            print("Downloading "+ curr_rad_obj.key.split('/')[-1])
            nexrad_bucket.download_file(Key=curr_rad_obj.key, Filename=radar_out_filename)
        radfiles.append(radar_out_filename)

    return radfiles

@TaskGenerator
def get_terrain_data(topography_file, projection, lonmin, lonmax, latmin, latmax, pad_deg=0.5):
    '''
    Load in the terrain data and convert the lat/lon coordinates to projection coordinates
    '''
    #load in topo dataset and plot
    topods = Dataset(topography_file)
    #these values need to be padded to avoid cutting off the terrain on the edges
    pad_deg = 0.5
    padlonmin = lonmin-pad_deg
    padlonmax = lonmax+pad_deg
    padlatmin = latmin-pad_deg
    padlatmax = latmax+pad_deg


    terlonmin = np.argmin(np.abs(np.array(topods.variables['x'])-(padlonmin)))
    terlonmax = np.argmin(np.abs(np.array(topods.variables['x'])-(padlonmax)))
    terlatmin = np.argmin(np.abs(np.array(topods.variables['y'])-(padlatmin)))
    terlatmax = np.argmin(np.abs(np.array(topods.variables['y'])-(padlatmax)))
    llon, llat = np.meshgrid(topods.variables['x'][terlonmin:terlonmax]
                             , topods.variables['y'][terlatmin:terlatmax])
    all_pts = projection.transform_points(ccrs.PlateCarree(), llon, llat)
    x = all_pts[:,:,0]
    y = all_pts[:,:,1]
    return x, y, np.array(topods.variables['z'][terlatmin:terlatmax, terlonmin:terlonmax])

@TaskGenerator
def load_metar_data(metar_dir,  start_datetime, end_datetime, states=['co', 'wy', 'ne'], pad_dt = datetime.timedelta(hours=1)):
    '''Load in the METAR data from the metar_dir
    Loads in individual metar files per state from the IEM database and trims only to the time of interest +/- the pad. 
    Note that this script will look for metar files named state_asos.txt in metar_dir. 
    '''
    state_dfs = list()
    for state in states:
        print(state)
        ssp = pd.read_csv(metar_dir+state+'_asos.txt', 
                parse_dates=['valid'], comment='#', on_bad_lines='skip')
        station_during_valid_time_sel = np.logical_and((ssp['valid']>(start_datetime-datetime.timedelta(hours=1))),
                                        (ssp['valid']<(end_datetime+datetime.timedelta(hours=1))))
        station_during_valid_time = ssp[station_during_valid_time_sel]
        state_dfs.append(station_during_valid_time)
        #df = df.append(station_during_valid_time, ignore_index=True)
    combined_df = pd.concat(state_dfs)
    return pd.concat(state_dfs)


# Download radar data from AWS S3 if not already downloaded
radfiles = get_radar_data(radarname, start_datetime, end_datetime, download_dir)
# Open up the AWS NEXRAD Level 2 resource

# Load in sondes 
sonde_list = glob.glob(sonde_dir+'/*')
sondes = list()
for sonde in sonde_list:
    print(sonde)
    sondes.append((parse_sonde(sonde), sonde.split('/')[-1]))


# Uncomment to plot the campus weather station. I had trouble downloading this data
'''
campuswxstn = pd.read_table('/Users/sfreeman/Documents/Research/c3x_radar_data/campus_wx_stn.txt',sep="\s*", header=(0), 
            parse_dates=[['Date','Time']])

campuswxstn['UTC_date'] = campuswxstn['Date_Time'].apply(lambda x: x.replace(tzinfo=pytz.timezone('America/Denver')).astimezone(pytz.utc))
'''
proj = ccrs.LambertConformal(central_longitude=-104, central_latitude=40
                                ,standard_parallels=[35])

terrain_x, terrain_y, terrain_data = jug.iteratetask(get_terrain_data(topography_file, projection=proj,
                                            lonmin=lonmin, lonmax=lonmax, latmin=latmin, latmax=latmax),3)
terrain_data = {'x': terrain_x, 'y':terrain_y, 'z': terrain_data}
all_metar_data = load_metar_data(metar_dir, start_datetime=start_datetime, end_datetime=end_datetime)
for radfilenum, filename in enumerate(jug.bvalue(radfiles)):
    plot_radar_data(filename, metar_data = all_metar_data, sondes = sondes, terrain_data=terrain_data)








