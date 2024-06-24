import matplotlib.pyplot as plt
import copy
import numpy as np
import rasterio
import cv2
import pandas as pd
import skimage.measure
import os


########################################################################################################################
# physical constants & parameters #
# Parameters
# elevation_change_rate = 5   # Example elevation change rate in meters/year
lake_level = 179      # lake water level in meters above sea level
res = 1
# Constants (replace these with appropriate values)
ice_density = 913           # Ice density in kg/m³
water_density = 1000         # fresh-water density in kg/m³
gravity = 9.81               # Acceleration due to gravity in m/s²
resolution = 100              # Pixel size in Meters
retreat_rate = 100              # Frontal retreat rate in meters per year

# Define the number of years and the interval for plotting
num_years = 79  # 2022 to 2100, 79 years
plot_interval = 1 # this defines how many years you would like to plot

factor = 1# this lets you control the factor to increase the surface lowering based on the surface lowering raster // ONLY IF FIXED ELEVATION CHANGE RATE
########################################################################################################################

# Data folders
workDir = 'C:/Users/....'
output_dir = 'C:/Users/....'
readme_path = f'{output_dir}README.txt'

# Directory to save the plots and GeoTIFFs
#output_dir = (f"{output_dir}/removed20192022dhdhtx2")  # Replace with your desired output directory

# load data maps
elevation_rast = rasterio.open(workDir + '/elevation_TDX_2023.tif')   #elevation_file here
thickness_rast = rasterio.open(workDir + '/2023updated_thickness_2nd_step.tif')  # ice thickness raster here
outlines_rast = rasterio.open(workDir + '/outlines_2023.tif')  #outlines here
dhdt_rast = rasterio.open(workDir + '/100mBin_min180tomax2280.tif') #dhdt elev binned tif here (optional)
bedrock_rast = rasterio.open(workDir + '/bedrock.tif') #bedrock raster here (optional)
lakelayer = rasterio.open(workDir + '/lakelayer.tif') #lake layer here (optional)
elevation_change_rates = pd.read_csv('C:/Users/.... .csv') #dhdt elev binned csv file here


#########################################################################################################################################################################################
#### Script starts here:

elevation_change_rates = elevation_change_rates.sort_values(by='Elevation', ascending=False)


# get lat and lon von axis
left, bottom, right, top = elevation_rast.bounds[0:4]
#small_extent = [624500, 4400000, 641500, 4410033]
(res_lat, res_lon) = elevation_rast.res
lat_range = np.arange(left, right, res_lat)
lon_range = np.arange(top, bottom, res_lon)

xv, yv = np.meshgrid(lat_range, lon_range)

elevation = elevation_rast.read(1).astype(np.float32)
thickness = thickness_rast.read(1).astype(np.float32)
outlines = outlines_rast.read(1).astype(np.float32)
bedrock_bathy = bedrock_rast.read(1).astype(np.float32)
lakelayer = lakelayer.read(1).astype(np.float32)
dhdht= dhdt_rast.read(1).astype(np.float32)
dhdht=dhdht*-1



# create a lake layer that has 'None' outside the elevation map
lake_surface = copy.copy(elevation)

lake_surface[elevation >= 0] = lake_level
lake_surface[elevation <= 0] = None


# compute bedrock and glacier surface
elevation[elevation <= 0] = None
thickness[thickness <= 0] = 0
bedrock = elevation - thickness
glacier_surface = copy.copy(elevation)
glacier_surface[outlines == 0] = None
thickness[outlines == 0] = None

# crop the lake layer to the area where bedrock is
lake_surface[bedrock > lake_level] = None

# Create a binary lake mask (1 for lake presence, 0 for no lake)
lake_mask_binary = (lake_surface == lake_level).astype(int)

# Initialize a list to store glacier surfaces for each year
glacier_surfaces = []

# Create a colormap that makes 0 values transparent
cmap = plt.get_cmap('viridis')
cmap.set_bad(alpha=0)  # Set the alpha (transparency) value for masked pixels (0 values)


# #############################################################################################
# Fill Gaps in Lake
# #############################################################################################
# get inverted ocean mask
lake_mask = np.copy(lake_mask_binary)
lake_mask = np.invert(lake_mask)
labeled_image, num_cluster = skimage.measure.label(lake_mask, connectivity=1, return_num=True)

cluster_size = np.zeros(num_cluster + 1)
for cluster_label in range(1, num_cluster + 1):
    cluster = labeled_image == cluster_label
    cluster_size[cluster_label] = cluster.sum()

final_cluster = cluster_size.argmax()

# create map of the gaps in lake area
gaps_mask = np.zeros_like(labeled_image)
gaps_mask[labeled_image >= 1] = 1
gaps_mask[labeled_image == final_cluster] = 0
# fill gaps
lake_mask_binary[gaps_mask == 1] = 1

# Frontal detection by array padding
# this function calculates the ice front for each iteration
'''def extract_front(glacier_surface, lake_mask_binary):
    # Create a mask for areas where no thickness data is present
    is_no_thickness_mask = np.isnan(glacier_surface).astype(np.uint8)

    # Combine the lake mask and no thickness mask
    final_lake_mask = lake_mask_binary * is_no_thickness_mask

    ICE_LAKE_mask = copy.copy(lake_mask_binary).astype(np.uint8)

    ICE_LAKE_mask = cv2.dilate(ICE_LAKE_mask, np.ones((3, 3), np.uint8), iterations=1)

    ICE_LAKE_mask[is_no_thickness_mask != 1] = 2
    #
    #ICE_LAKE_mask_dilated = cv2.dilate(ICE_LAKE_mask, np.ones((3, 3), np.uint8), iterations=1)

    kernel_erode = np.array([[0,1,0], [1, 1, 1], [0,1 , 0]], np.uint8)
    ICE_LAKE_mask_eroded = cv2.erode(ICE_LAKE_mask, kernel_erode, iterations=1)

    #kernel_dilate = np.array([[0,1,0], [0, 1, 1], [0,1 , 0]], np.uint8)
    #ICE_LAKE_mask_dilated = cv2.dilate(ICE_LAKE_mask_eroded, kernel_dilate, iterations=1)
    #
    mask_mi = np.pad(ICE_LAKE_mask_eroded, ((1, 1), (1, 1)), mode='constant')
    mask_le = np.pad(ICE_LAKE_mask_eroded, ((1, 1), (0, 2)), mode='constant')
    mask_ri = np.pad(ICE_LAKE_mask_eroded, ((1, 1), (2, 0)), mode='constant')
    mask_do = np.pad(ICE_LAKE_mask_eroded, ((0, 2), (1, 1)), mode='constant')
    mask_up = np.pad(ICE_LAKE_mask_eroded, ((2, 0), (1, 1)), mode='constant')

    #front = np.logical_and(mask_mi == 2, np.logical_or.reduce((mask_do == 1, mask_up == 1, mask_ri == 1, mask_le == 1)))
    front = np.logical_and(mask_mi == 1, np.logical_or.reduce((mask_do == 2, mask_up == 2, mask_ri == 2, mask_le == 2))) # moves it towards the water
    front = front[1:-1, 1:-1].astype(np.uint8)

    front_dilated = cv2.dilate(front, np.ones((3, 3), np.uint8), iterations=0)
 
    return front_dilated'''


# Countour frontal detection
def extract_front(glacier_surface, lake_mask_binary):
    glacier = np.where(glacier_surface >= 1, 1, 0)
                #glacier = cv2.cvtColor(glacier, cv2.COLOR_BGR2GRAY)
    glacier = glacier.astype(np.uint8)
    contours, hierarchy = cv2.findContours(glacier, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    zero_image = np.zeros(glacier.shape)
    contour_image = cv2.drawContours(zero_image, contours, -1, (255, 255, 255)).astype(np.uint8)
    front = np.logical_and(contour_image, lake_mask_binary)
    return front

# Create a contour plot using the "outlines" array
contours = plt.contour(outlines, levels=[0.5], colors='black', linewidths=1)

# Create a list to store results for each year
results = []

for year in range(2023, 2101):
    '''# Remove an elevation change rate field from the raster(s): #this is not needed if a .csv file is used 
    glacier_surface -= dhdht * factor
    thickness -= dhdht * factor'''
    


    # Iterate over each pixel in the raster including elevation feedback from dh/dt
    for i in range(glacier_surface.shape[0]):
        for j in range(glacier_surface.shape[1]):
            # Get the current elevation of the pixel
            current_elevation = glacier_surface[i, j]

            # Find the first row in the DataFrame where the elevation is greater than the current elevation
            row_index = (elevation_change_rates['Elevation'] > current_elevation).idxmax()

            # If there is a row found, get the corresponding change rate
            if row_index > 0:
                change_rate = elevation_change_rates.loc[row_index, 'Change Rate']

                # Make the change rate positive before applying it
                positive_change_rate = abs(change_rate)

                # Apply the positive change rate to update the glacier_surface and thickness arrays
                glacier_surface[i, j] -= dhdht[i, j] * positive_change_rate
                thickness[i, j] -= dhdht[i, j] * positive_change_rate

    # call extract front function
    front = extract_front(glacier_surface, lake_mask_binary)
    # Calculate the overlap between the front_dilated and elevation masks
    front_mask = np.logical_and(front, glacier_surface)
    # Convert boolean array to integer (0 for False, 1 for True)
    front_mask_int = front_mask.astype('uint8')

   # Define the output file path
    output_file = (f'C:/Users/ .... /icefront_{retreat_rate}_{year}_dialate.tif')


    # Save the glacier surface as a GeoTIFF
    with rasterio.open(output_file, 'w', driver='GTiff', height=glacier_surface.shape[0],
                       width=glacier_surface.shape[1], count=1, dtype='uint8',
                       crs=elevation_rast.crs, transform=elevation_rast.transform) as dst:
        dst.write(front_mask_int, 1)

    #Find indices where front_mask is True
    indices = np.where(front_mask)

    # Open a text file for writing
    with open(f'{output_dir}forced_frontal_loss.txt', 'a') as file:
        # Iterate over the indices and access corresponding elements of glacier_surface
        for i in range(len(indices[0])):

            row_idx = indices[0][i]
            col_idx = indices[1][i]


            # Check if thickness value is not NaN
            if not np.isnan(thickness[row_idx, col_idx]):
                # Calculate the volume to subtract for the current pixel individually
                #ice_column = thickness[row_idx, col_idx] * resolution * resolution
                slice_volume = thickness[row_idx, col_idx] * resolution * retreat_rate
                cube_base_area = resolution * resolution

                # Subtract the calculated volume from the current pixel
                height_to_remove = slice_volume / cube_base_area

                # Write output to the file
                file.write(f"Year: {year}\n")
                file.write(f"Height to Remove: {height_to_remove}\n")
                file.write(f"Before Glacier Surface [{row_idx}, {col_idx}]: {glacier_surface[row_idx, col_idx]}\n")
                glacier_surface[row_idx, col_idx] -= height_to_remove
                file.write(f"After Glacier Surface [{row_idx}, {col_idx}]: {glacier_surface[row_idx, col_idx]}\n")
            else:
                # Handle the case where thickness is NaN
                pass

    # hf = height of flotation
    # water depth = lake_surface - bedrock
    #
    hf = np.where(bedrock > lake_surface, None, bedrock + (water_density / ice_density) * (lake_surface - bedrock))

    # Convert the hf array to a NumPy array
    hf = np.array(hf, dtype=float)
    hf[hf == None] = np.nan  # Replace None values with NaN
    hab = glacier_surface - hf

    # Set pixels to None where hf is smaller than 0 or elevation is below sea level or simply below bedrock
    glacier_surface[(hab < 0) | (glacier_surface < 179) | (glacier_surface < bedrock)] = None

    # Append a copy of the glacier surface to the list for plotting
    glacier_surfaces.append(np.ma.masked_equal(glacier_surface, None))  # Mask None values

    # Plot the glacier surface and save as a PNG every plot_interval years
    if year % plot_interval == 0:
        plt.imshow(glacier_surfaces[-1], cmap=cmap,)# extent=[left, right, bottom, top]
        plt.title(f'Glacier Surface in {year}')
        plt.colorbar(label='Elevation (m)')

        # Create a contour plot using the "outlines" array
        #contours = plt.contour(lat_range, lon_range, outlines, levels=[0.5], colors='black', linewidths=1 )
        contours = plt.contour( outlines, levels=[0.5], colors='black', linewidths=1 )

        # Save the plot with the year in the filename
        plot_filename = f'{output_dir}glacier_surface_{year}.png'
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to avoid displaying it

        plt.imshow(front_mask, cmap='viridis', extent=[left, right, bottom, top])
        plt.title(f'Glacier Front in {year}')
        plt.colorbar(label='Lake Presence (1) / No Lake (0)')

        # Save the plot with the year in the filename
        plot_filename = f'{output_dir}glacier_front_{year}.png'
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to avoid displaying it

        # Save the glacier surface as a GeoTIFF
        tiff_filename = f'{output_dir}glacier_surface_rast{year}.tif'
        with rasterio.open(tiff_filename, 'w', driver='GTiff', height=glacier_surface.shape[0],
                          width=glacier_surface.shape[1], count=1, dtype=str(glacier_surface.dtype),
                          crs=elevation_rast.crs, transform=elevation_rast.transform) as dst:
            dst.write(glacier_surface, 1)



#Writing a readme file in the output dir to save all parameters:
with open(readme_path, 'w') as readme_file:
    readme_file.write("#" * 50 + "\n")
    readme_file.write("# Physical Constants & Parameters #\n")
    readme_file.write("# Parameters\n")
    readme_file.write(f"lake_level = {lake_level}      # lake water level in meters above sea level\n")
    readme_file.write(f"res = {res}\n")
    readme_file.write("# Constants (replace these with appropriate values)\n")
    readme_file.write(f"ice_density = {ice_density}           # Ice density in kg/m³\n")
    readme_file.write(f"water_density = {water_density}         # fresh-water density in kg/m³\n")
    readme_file.write(f"gravity = {gravity}               # Acceleration due to gravity in m/s²\n")
    readme_file.write(f"resolution = {resolution}              # Pixel size in Meters\n")
    readme_file.write(f"retreat_rate = {retreat_rate}              # Frontal retreat rate in meters per year\n")
    readme_file.write("# Define the number of years and the interval for plotting\n")
    readme_file.write(f"num_years = {num_years}  # 2022 to 2100, 79 years\n")
    readme_file.write(f"plot_interval = {plot_interval} # this defines how many years you would like to plot\n")
    readme_file.write("\n")
    readme_file.write(f"factor = {factor}# this lets you control the factor to increase the surface lowering based on the surface lowering raster\n")
    readme_file.write("#" * 50 + "\n")
    readme_file.write("# Data Folders #\n")
    readme_file.write(f"Work Directory: {workDir}\n")
    readme_file.write(f"Output Directory: {output_dir}\n")

    readme_file.write("# Data Maps #\n")
    readme_file.write(f"Elevation Raster: {elevation_rast}\n")
    readme_file.write(f"Thickness Raster: {thickness_rast}\n")
    readme_file.write(f"Outlines Raster: {outlines_rast}\n")
    readme_file.write(f"Dh/Dt Raster: {dhdt_rast}\n")

