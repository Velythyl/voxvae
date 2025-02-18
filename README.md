# Vox Autoencoder With Voxel IDs

### Dataset shape

Datasets should consist in a JSON that associates names to segmented point clouds. The JSONs are of this shape:

```json

{
  <`>FULL<` or `>ORIGINAL XML<`>: {
    "pcd_points": <list of points>,
    "pcd_colors": <list of colors; should be one single color in most cases>,
    "color": "all"
  },
  <part name>: {
    "color": <the one single color that belongs to <part name>,
   },
  ...
}
```

There are two special part names. They are:

- `>FULL<` is the concatenation of all normal parts
- `>ORIGINAL XML<` is the non-segmented point cloud, which we ignore
