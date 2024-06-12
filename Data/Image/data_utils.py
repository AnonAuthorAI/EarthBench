import numpy as np
import os
import requests as r
import tqdm
import rasterio
from rasterio.transform import xy
from rasterio.crs import CRS
from copy import copy
from skimage import io
import matplotlib.pyplot as plt
import pyproj


# GLOBAL VARIABLES
LAT_SPACE = np.arange(-90, 90.25, 0.25)
LON_SPACE = np.arange(0, 360.25, 0.25)
BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07']
TILE_LAT_SPAN = 1.3132319999999993
TILE_LON_SPAN = 1.0077099999999959
TILE_TIF_RESOLUTION = (3660, 3660)

def bbox_contain(b1, b2):
    """Return whether b2 strictly contains b1"""
    return b1[0] >= b2[0] and b1[1] >= b2[1] and b1[2] <= b2[2] and b1[3] <= b2[3]

def download_tif(url, dir_path:str = "./hls_data"):
    """
    Download a .tif file from a URL to a specified directory.

    Parameters:
    - url (str): The URL of the .tif file to download.
    - dir_path (str): The directory path to save the downloaded file.

    Returns:
    str: The path to the saved .tif file.
    """
    assert os.path.isdir(dir_path), f"Directory '{dir_path}' does not exist."
    
    # Get filename from URL and append to dir_path
    filename = url.split('/')[-1]
    if filename[-4:] != '.tif':
        raise ValueError(f"URL {url} does not point to a .tif file.")
    save_path = os.path.join(dir_path, filename)

    # Check if file already exists
    if os.path.isfile(save_path):
        pass
    # If not, download file
    else:
        response = r.get(url)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
        else:
            raise RuntimeError(f"Error HTTP Code {response.status_code} when downloading file from {url}.")
    
    return save_path

def transform_to_relative_coord(bbox, tile_bbox, tile_image_resolution:tuple = (3660, 3660)):
    """
    Transform bbox to relative coordinates to tile_bbox.

    Parameters:
    - bbox (list): Bounding box coordinates [lat_min, lon_min, lat_max, lon_max].
    - tile_bbox (list): Tile bounding box coordinates [lat_min, lon_min, lat_max, lon_max].
    - tile_image_resolution (tuple): Resolution of the tile image.

    Returns:
    np.array: Transformed relative bbox coordinates.
    """
    assert bbox_contain(bbox, tile_bbox), f"bbox {bbox} is not contained in tile_bbox {tile_bbox}"
    span_lat, span_lon = (tile_bbox[2] - tile_bbox[0]), (tile_bbox[3] - tile_bbox[1])
    base_lat, base_lon = tile_bbox[0], tile_bbox[3]
    res_lat, res_lon = tile_image_resolution
    x1 = (bbox[0] - base_lat) / span_lat * res_lat
    x2 = (bbox[2] - base_lat) / span_lat * res_lat
    y1 = (base_lon - bbox[3]) / span_lon * res_lon  # due to the inversed y-axis
    y2 = (base_lon - bbox[1]) / span_lon * res_lon  # due to the inversed y-axis
    res_bbox = np.array([x1, y1, x2, y2]).astype(int)
    return res_bbox

def adjust_bbox_for_partially_observed_tiles(hls_item, debug=False):
    """
    Fix the bbox of partially observed tile.

    Parameters:
    - hls_item (dict): The HLS item dictionary.
    - debug (bool): Whether to print debug information.

    Returns:
    dict: The updated HLS item with fixed bbox.
    """
    fmask = download_tif(hls_item['assets']['Fmask']['href'], "./hls_data/fmask/")
    with rasterio.open(fmask) as mask_tif:
        mask_image = mask_tif.read()[0,:,:]
        mask_valid = mask_image < 255
    corner_anchor = {
        'ul': mask_valid[0, 0],
        'll': mask_valid[-1, 0],
        'ur': mask_valid[0, -1],
        'lr': mask_valid[-1, -1],
    }
    # if all corners are valid, then the bbox is correct
    if sum(corner_anchor.values()) == 4:
        return hls_item

    org_bbox = hls_item['bbox']
    for corner, anchor_flag in corner_anchor.items():
        if anchor_flag:
            if corner == 'ul':
                anchor_lonlat = (org_bbox[0], org_bbox[3])
                offset_lonlat = (0, -TILE_LON_SPAN)
            elif corner == 'll':
                anchor_lonlat = (org_bbox[0], org_bbox[1])
                offset_lonlat = (0, 0)
            elif corner == 'ur':
                anchor_lonlat = (org_bbox[2], org_bbox[3])
                offset_lonlat = (-TILE_LAT_SPAN, -TILE_LON_SPAN)
            elif corner == 'lr':
                anchor_lonlat = (org_bbox[2], org_bbox[1])
                offset_lonlat = (-TILE_LAT_SPAN, 0)
            break
    x1, y1 = (anchor_lonlat[0] + offset_lonlat[0], anchor_lonlat[1] + offset_lonlat[1])
    fixed_bbox = [x1, y1, x1 + TILE_LAT_SPAN, y1 + TILE_LON_SPAN]
    if debug:
        print(
            f"Anchor used: {corner.upper()}\n"
            f"Status:      {corner_anchor}\n"
            f"Org bbox:    {org_bbox}\n"
            f"Fix bbox:    {fixed_bbox}"
        )
        plt.suptitle(f"adjust_bbox_for_partially_observed_tiles() debug output")
        plt.imshow(mask_image)
        plt.show()
    hls_item['bbox'] = fixed_bbox
    return hls_item

def check_tile_has_full_image_for_bbox(
    hls_item, query_bbox, 
    tolerance=0.01, 
    debug=False
):
    """
    Check whether the tile has an image for the query bbox using the Fmask file.

    Parameters:
    - hls_item (dict): The HLS item dictionary.
    - query_bbox (list): The query bounding box coordinates [lat_min, lon_min, lat_max, lon_max].
    - tolerance (float): Tolerance for missing data.
    - debug (bool): Whether to print debug information.

    Returns:
    bool: True if the tile has an image for the query bbox, False otherwise.
    """
    # Adjust the tile bbox if the tile is partially observed
    hls_item = adjust_bbox_for_partially_observed_tiles(hls_item, debug=debug)
    tile_bbox = hls_item['bbox']
    # Check if the tile bbox contains the query bbox
    if not bbox_contain(query_bbox, hls_item['bbox']):
        return False

    # Download the Fmask file
    fmask_path = download_tif(hls_item['assets']['Fmask']['href'], "./hls_data/fmask/")
    with rasterio.open(fmask_path) as mask_tif:
        tif_metadata = mask_tif.profile
        # Get the image resolution
        tif_resolution = (tif_metadata['width'], tif_metadata['height'])
        # Locate the query bbox in the tile image
        image_bbox = transform_to_relative_coord(query_bbox, tile_bbox, tile_image_resolution=tif_resolution)
        # Get the subfmask for the query bbox
        subfmask = mask_tif.read(
            window=((image_bbox[1], image_bbox[3]), (image_bbox[0], image_bbox[2]))
        )  
        # Calculate the empty ratio
        empty_ratio = (subfmask == 255).sum() / subfmask.size

        if debug:
            fig, (ax0, ax1, ax2) = plt.subplots(1,3,figsize=(15,5))
            image = io.imread(hls_item['assets']['browse']['href'])
            ax0.imshow(image)
            ax0.set_title(f"tile_bbox: {tile_bbox}")
            fmask_image = mask_tif.read()
            ax1.imshow(fmask_image[0,:,:])
            ax1.plot(
                [image_bbox[0], image_bbox[0], image_bbox[2], image_bbox[2], image_bbox[0]], 
                [image_bbox[1], image_bbox[3], image_bbox[3], image_bbox[1], image_bbox[1]], 
                color='r',
            )
            ax1.set_title(f"Fmask image")
            ax2.imshow(subfmask[0,:,:])
            ax2.set_title(f"subfmask empty_ratio: {empty_ratio}")
            plt.suptitle(f"check_tile_has_full_image_for_bbox() debug output")
            plt.tight_layout()
            plt.show()
        
    return empty_ratio < tolerance

def query_hls_image(
    lonlat: tuple,
    span: tuple = (0.125, 0.125),
    time_period: tuple = ("05-01", "08-01"),
    years: list = [2023, 2022, 2021],
    collections: list = ['HLSS30.v2.0', 'HLSL30.v2.0'],
    cloud_cover_tolerance: int = 5,
    missing_data_tolerance: int = 1,
    n_response: int = 3,
    debug: bool = False,
    verbose: bool = True,
):
    """
    Query HLS images based on specified parameters.
    HLS S30 Start Date is: 2015-12-01T00:00:00.000Z
    HLS L30 Start Date is: 2013-05-01T00:00:00.000Z

    Parameters:
    - lonlat (tuple): Latitude and longitude coordinates (lat, lon).
    - span (tuple): Latitude and longitude span (lat_span, lon_span).
    - time_period (tuple): Start and end date of the time period (start_date, end_date).
    - years (list): List of years to search for.
    - collections (list): List of HLS collections to search in.
    - cloud_cover_tolerance (int): Maximum allowed cloud cover percentage.
    - missing_data_tolerance (int): Maximum allowed missing data percentage.
    - n_response (int): Number of desired responses.
    - debug (bool): Whether to print debug information.

    Returns:
    list: List of HLS items matching the query parameters.
    """
    assert len(lonlat) == 2, f"lonlat should be a tuple of length 2, but got {lonlat}"
    assert len(span) == 2, f"span should be a tuple of length 2, but got {span}"
    assert len(time_period) == 2, f"time_period should be a tuple of length 2, but got {time_period}"
    assert len(years) > 0, f"years should be a list of length > 0, but got {years}"
    assert len(collections) > 0, f"collections should be a list of length > 0, but got {collections}"
    assert cloud_cover_tolerance >= 0 and cloud_cover_tolerance <= 100, \
        f"cloud_cover_tolerance should be in [0, 100], but got {cloud_cover_tolerance}"
    assert missing_data_tolerance >= 0 and missing_data_tolerance <= 100, \
        f"missing_data_tolerance should be in [0, 100], but got {missing_data_tolerance}"
    assert n_response > 0, f"n_response should be > 0, but got {n_response}"

    # initialize the search endpoint and query parameters
    lp_search_endpoint = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search"
    years = sorted(years, reverse=True)
    query_timestamps = [
        f"{year}-{time_period[0]}T00:00:00Z/{year}-{time_period[1]}T23:59:59Z"
        for year in years
    ]
    query_bbox = [lonlat[0]-span[0], lonlat[1]-span[1], lonlat[0]+span[0], lonlat[1]+span[1]]
    if verbose:
        print(
            f"==================== QUERY INFO ====================\n"
            f"lat-lon:                {lonlat}\n"
            f"span:                   {span}\n"
            f"generated bbox:         {query_bbox}\n"
            f"time_period:            {time_period}\n"
            f"years:                  {years}\n"
            f"collections:            {collections}\n" 
            f"cloud_cover_tolerance:  {cloud_cover_tolerance}\n"
            f"n_response:             {n_response}\n"
            f"====================================================\n"
    )

    # Start from the most recent date and work backwards until we find an image
    all_hls_items = []
    for query_timestamp in query_timestamps:

        search_params = {
            "bbox": query_bbox, # Lat/lon Bounding Box (lower left, upper right)
            'datetime': query_timestamp, # Date range
            "collections": collections, # Collection(s) to search
            'limit': 100, # Number of items returned
        }
        if verbose:
            print(f"Searching for images from time period {query_timestamp}...")
            print(f"Search parameters: {search_params}")
        hls_items = r.post(lp_search_endpoint, json=search_params).json()['features']
        if verbose:
            print(f"{len(hls_items)} items returned from NASA earth data search.")

        # Filter the hls_items by cloud coverage and bbox containment
        hls_items = [hls_item for hls_item in hls_items if hls_item['properties']['eo:cloud_cover'] <= cloud_cover_tolerance]
        if verbose:
            print(f"{len(hls_items)} items left after filtering by cloud coverage.")
        hls_items = [hls_item for hls_item in hls_items if bbox_contain(query_bbox, hls_item['bbox'])]
        if verbose:
            print(f"{len(hls_items)} items left after filtering by bbox containment.")
        valid_hls_items = []
        for hls_item in tqdm.tqdm(hls_items, desc=f"Validating {len(hls_items)} items", disable=not verbose):
            # check whether the tile has image for the query bbox using the Fmask file
            if check_tile_has_full_image_for_bbox(
                hls_item, query_bbox,
                tolerance=missing_data_tolerance/100,
                debug=debug,
            ):
                valid_hls_items.append(hls_item)
                if len(valid_hls_items) >= n_response:
                    if verbose:
                        print(f"Found {len(valid_hls_items)} valid items, early stopping.")
                    break

        # sort the valid_hls_items by cloud coverage
        valid_hls_items = sorted(valid_hls_items, key=lambda k: k['properties']['eo:cloud_cover'])
        
        # store the valid_hls_items
        all_hls_items.extend(valid_hls_items)

        # break if we have enough responses
        if len(all_hls_items) >= n_response:
            break
    
    # get the latest n_response items
    if len(all_hls_items) >= n_response:
        responses = all_hls_items[:n_response]
    elif len(all_hls_items) > 0:
        if verbose:
            print(f"Only {len(all_hls_items)} images found for the given query.")
        responses = all_hls_items
    else:
        raise Exception("No images found for the given query.")
    
    if debug:
        for i, s in enumerate(responses):
            print(
                f"({i}) {s['id']} | BBOX: {s['bbox']} | Date: {s['properties']['datetime']} | Cloud Cover: {s['properties']['eo:cloud_cover']}"
            )
        base_figsize = 4
        fig, axes = plt.subplots(1, len(responses), figsize=(base_figsize*len(responses), base_figsize))
        for i, response in enumerate(responses):
            rgb_image = io.imread(response['assets']['browse']['href'])
            image_res = (rgb_image.shape[0], rgb_image.shape[1])
            rgb_image_bbox = transform_to_relative_coord(query_bbox, response['bbox'], tile_image_resolution=image_res)
            axes[i].imshow(rgb_image)
            axes[i].plot(
                [rgb_image_bbox[0], rgb_image_bbox[0], rgb_image_bbox[2], rgb_image_bbox[2], rgb_image_bbox[0]], 
                [rgb_image_bbox[1], rgb_image_bbox[3], rgb_image_bbox[3], rgb_image_bbox[1], rgb_image_bbox[1]], 
                color='r',
            )
            axes[i].set_title(f"({i}) {response['id']}")
        plt.suptitle(f"query_hls_image() debug output: Response Images")
        plt.tight_layout()
        plt.show()

    return responses

def plot_tif_rgb(tif_path):
    with rasterio.open(tif_path) as tif:
        image = tif.read()
        rgb_image = np.stack([image[2,:,:], image[1,:,:], image[0,:,:]], axis=2)
        rgb_image = rgb_image / 2000
        print(f"Plotting RGB image of {tif_path}.")
        plt.imshow(rgb_image)
        plt.show()

def get_lonlat_googlemap_link(lonlat, zoom=14):
    """
    Generate a Google Maps URL pointing to the satellite view of a given latitude and longitude.

    Parameters:
    lonlat (tuple): Tuple of longitude and latitude coordinates.
    zoom (int): Zoom level for the map. Default is 15.

    Returns:
    str: URL string to the Google Maps satellite view.
    """
    lon, lat = lonlat
    base_url = "https://www.google.com/maps"
    # Compose the URL with parameters for latitude, longitude, and satellite view
    url = f"{base_url}/@{lat},{lon},{zoom}z/data=!3m1!1e3"
    return url

def get_tif_lonlat(tif_path, debug=False):
    with rasterio.open(tif_path) as tif:
        # get the affine transformation matrix
        transform = tif.transform
        # get the image width and height
        width = tif.width
        height = tif.height
        # get the coordinate reference system
        crs = tif.crs

        # compute the pixel coordinates of the top-left and bottom-right corners
        top_left_pixel = (0, 0)
        bottom_right_pixel = (width, height)

        # transform the pixel coordinates to image coordinates
        top_left_coords = xy(transform, *top_left_pixel, offset="ul")
        bottom_right_coords = xy(transform, *bottom_right_pixel, offset="lr")

        if debug:
            print(f"Top left coordinates (image CRS): {top_left_coords}")
            print(f"Bottom right coordinates (image CRS): {bottom_right_coords}")

        # transform the image coordinates to longitude and latitude
        if crs.is_geographic:
            top_left_lonlat = top_left_coords
            bottom_right_lonlat = bottom_right_coords
        else:
            transformer = pyproj.Transformer.from_crs(
                crs, CRS.from_epsg(4326), always_xy=True
            )
            top_left_lonlat = transformer.transform(*top_left_coords)
            bottom_right_lonlat = transformer.transform(*bottom_right_coords)
        if debug:
            print(f"Top left coordinates (longitude, latitude): {top_left_lonlat}")
            print(
                f"Bottom right coordinates (longitude, latitude): {bottom_right_lonlat}"
            )
            center_lonlat = (
                (top_left_lonlat[0] + bottom_right_lonlat[0]) / 2,
                (top_left_lonlat[1] + bottom_right_lonlat[1]) / 2,
            )
            print(f"Center coordinates (longitude, latitude): {center_lonlat}")
            googlemap_link = get_lonlat_googlemap_link(center_lonlat)
            print(f"Google Map (for validation): {googlemap_link}")

    return top_left_lonlat, bottom_right_lonlat