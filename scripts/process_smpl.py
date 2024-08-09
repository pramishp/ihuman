import os

import bpy
import bmesh
import numpy as np
import sys

import os
import bpy
import bmesh
import numpy as np
import sys

import numpy as np
from scipy.spatial import cKDTree


# Load an FBX file
def load_fbx(file_path):
    bpy.ops.import_scene.fbx(filepath=file_path)


# Convert quads to tris for all meshes
def convert_quads_to_tris():
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.quads_convert_to_tris()
            bpy.ops.object.mode_set(mode='OBJECT')


# Upsample mesh using Subdivision Surface modifier
def upsample_mesh(subdivision_levels):
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            remove_shape_keys(obj)
            modifier = obj.modifiers.new(name='Subsurf', type='SUBSURF')
            modifier.levels = subdivision_levels
            modifier.render_levels = subdivision_levels
            bpy.ops.object.modifier_apply(modifier='Subsurf')


def remove_shape_keys(obj):
    if obj.data.shape_keys:
        obj.shape_key_clear()


def subdivide_vertex_group(obj, group_names, subdivision_factor, use_catmull_clark=False):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Get the vertex group indices
    group_indices = {obj.vertex_groups[group_name].index for group_name in group_names}

    # Create a bmesh from the active object's mesh
    bm = bmesh.from_edit_mesh(obj.data)

    # Use a set to store unique edges to be subdivided
    edges_to_subdivide = set()

    # Iterate over the faces to find those associated with the vertex groups
    for face in bm.faces:
        # Check if any vertex in the face belongs to the specified vertex groups
        if any(vertex in face.verts for vertex in bm.verts if
               any(group in group_indices for group in [vg.group for vg in obj.data.vertices[vertex.index].groups])):
            face.select = True  # Select the face
            # Add the face's edges to the set, ensuring uniqueness
            edges_to_subdivide.update(face.edges)
        else:
            face.select = False  # Deselect the face

    # Use subdivide_edges for both Loop and Catmull-Clark subdivision
    if use_catmull_clark:
        # Parameters adjusted for Catmull-Clark effect
        bmesh.ops.subdivide_edges(bm, edges=list(edges_to_subdivide), cuts=subdivision_factor, use_grid_fill=True,
                                  use_smooth=True, smooth_falloff='INVERSE_SQUARE')
    else:
        # Parameters for Loop subdivision
        bmesh.ops.subdivide_edges(bm, edges=list(edges_to_subdivide), cuts=subdivision_factor, use_grid_fill=False)

    # Update the mesh with the new subdivision
    bmesh.update_edit_mesh(obj.data)

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')


def remove_shape_keys(obj):
    if obj.data.shape_keys:
        obj.shape_key_clear()
        # obj.data.shape_keys.animation_data_clear()
        # obj.data.shape_keys.key_blocks.clear()


def duplicate_without_shape_keys(obj):
    obj.select_set(True)
    bpy.ops.object.duplicate(linked=False, mode='TRANSLATION')
    dup_obj = bpy.context.selected_objects[0]  # Assuming the duplicate is the active object
    remove_shape_keys(dup_obj)
    return dup_obj


# Save joint information
def save_joints(output_path):
    # Make sure the correct objects are selected
    armature = bpy.data.objects['Armature']  # Replace with your armature name if different
    mesh_obj = bpy.data.objects['m_avg']  # Replace with your mesh name if different

    # Ensure the armature and mesh objects are correct
    if armature.type == 'ARMATURE' and mesh_obj.type == 'MESH':
        # The names of the bones in the order you need them
        bone_names = [
            'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
            'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
            'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
            'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
        ]

        # Initialize an array to store joint locations
        joint_locations = np.zeros((len(bone_names), 3), dtype=np.float32)

        # Calculate the inverse of the mesh object's world matrix
        mesh_inv_world_matrix = mesh_obj.matrix_world.inverted()

        # Get the pose bones
        pose_bones = armature.pose.bones

        # Iterate through each bone name and get its head position in local space
        for i, bone_name in enumerate(bone_names):
            # Find the corresponding pose bone, prefixing bone names with 'm_avg_'
            pbone = pose_bones.get("m_avg_" + bone_name)  # Adjust the prefix as per your naming convention
            if pbone:
                # Convert the bone's head position to the mesh's local space
                joint_locations[i] = mesh_inv_world_matrix @ (armature.matrix_world @ pbone.head)
            else:
                print(f"Bone named '{bone_name}' not found in the armature.")

        # Save the joint locations to a text file
        np.savetxt(output_path, joint_locations, fmt='%.6f')

        print("Joint locations saved successfully.")
    else:
        print("Please select an armature and ensure a mesh object named 'm_avg' exists.")


# Save mesh data (vertices, triangles, UVs)
def save_mesh_data(vertices_path, triangles_path, uv_coords_path, uv_map2face_path):
    bpy.ops.object.mode_set(mode='OBJECT')
    obj = bpy.context.active_object
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    # Triangulate mesh
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    vertices = np.array([vert.co for vert in bm.verts])
    triangles = np.array([[vert.index for vert in face.verts] for face in bm.faces])
    unique_uvs, uv_index_dict, uv_map2face = [], {}, []
    if len(bm.loops.layers.uv) > 0:
        uv_layer = bm.loops.layers.uv.verify()
        for face_index, face in enumerate(bm.faces):
            for loop in face.loops:
                uv = loop[uv_layer].uv
                uv_tuple = (uv.x, uv.y)
                if uv_tuple not in uv_index_dict:
                    unique_uvs.append(uv_tuple)
                    uv_index = len(unique_uvs) - 1
                    uv_index_dict[uv_tuple] = uv_index
                    uv_map2face.append(face_index)
                else:
                    uv_index = uv_index_dict[uv_tuple]
                    uv_map2face[uv_index] = face_index
    bm.free()
    unique_uvs_array = np.array(unique_uvs)
    uv_map2face_array = np.array(uv_map2face)
    np.savetxt(vertices_path, vertices, fmt='%.6f')
    np.savetxt(triangles_path, triangles, fmt='%d')
    np.savetxt(uv_coords_path, unique_uvs_array, fmt='%.6f')
    np.savetxt(uv_map2face_path, uv_map2face_array, fmt='%d')


def parse_args():
    # Default values
    fbx_file_path = ''
    save_path_root = './'
    subdivision_levels = 2

    # Parse arguments passed to Blender after "--"
    argv = sys.argv
    if "--" not in argv:
        print("No arguments passed to script. Using default values.")
        return fbx_file_path, save_path_root, subdivision_levels

    args = argv[argv.index("--") + 1:]  # Get all arguments after "--"

    if '--root' in args:
        save_path_root_index = args.index('--root') + 1
        if save_path_root_index < len(args):
            save_path_root = args[save_path_root_index]

    if '--subdivision' in args:
        subdivision_index = args.index('--subdivision') + 1
        if subdivision_index < len(args):
            subdivision_levels = int(args[subdivision_index])

    if len(args) > 0:
        fbx_file_path = args[-1]  # Assume the first argument is the FBX file path

    return fbx_file_path, save_path_root, subdivision_levels



# Load the point clouds from text files
def load_point_cloud(file_path):
    return np.loadtxt(file_path, delimiter=' ')


# Align pc1 to pc2 (this is a simple translation alignment for demonstration)
def align_point_clouds(pc1, pc2):
    # Compute the centroids of each point cloud
    centroid_pc1 = np.mean(pc1, axis=0)
    centroid_pc2 = np.mean(pc2, axis=0)

    # Translate pc1 to pc2 by the difference of centroids
    aligned_pc1 = pc1 - centroid_pc1 + centroid_pc2
    return aligned_pc1


# Create an index mapping pc1 points to the nearest pc2 points
def create_mapping_index(aligned_pc1, pc2):
    # Build a KD-Tree for efficient nearest neighbor search
    tree = cKDTree(pc2)

    # Query the nearest neighbor of each point in pc1
    distances, indices = tree.query(aligned_pc1, k=1)
    return indices

if __name__ == "__main__":
    fbx_file_path, save_path_root, subdivision_levels = parse_args()
    print("\n\n\ Print Args \n\n")
    print(save_path_root)
    print(subdivision_levels)
    print(fbx_file_path)

    root = save_path_root
    os.makedirs(root, exist_ok=True)
    if not fbx_file_path:
        raise Exception("No FBX file path provided.")

    # Parameters
    # root = "/home/pramish/Downloads/smpl male GS init/from blender"
    # fbx_file_path = '/home/pramish/Downloads/smpl male GS init/SMPL_maya/basicModel_m_lbs_10_207_0_v1.0.2.fbx'
    # subdivision_levels = 3  # Change as needed

    # Execution
    print("Reading fbx from path ", fbx_file_path)
    load_fbx(fbx_file_path)
    convert_quads_to_tris()

    print("----------Saving Joints----------")
    save_joints(f"{root}/joint_locations.txt")

    print("--------- Sub Dividing head and hand ---------")
    # Assuming the last mesh object in the scene is the one we're interested in
    mesh_obj = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH'][-1]

    # Selective subdivision on specified vertex groups
    selective_subdivision_groups = ['m_avg_Head', 'm_avg_L_Hand', 'm_avg_R_Hand']
    # selective_subdivision_groups = ['m_avg_Head']
    # subdivide_vertex_group(mesh_obj, selective_subdivision_groups, 1, use_catmull_clark=True)  # '1' makes two faces out of one

    # print('Upsampling mesh with subdivision levels ', subdivision_levels)
    if subdivision_levels > 0:
        upsample_mesh(subdivision_levels)

    # Assuming the last mesh object in the scene is the one we're interested in
    bpy.context.view_layer.objects.active = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH'][-1]

    if not os.path.exists(root):
        os.mkdir(root)

    print("------Saving other mesh data------")
    save_mesh_data(f"{root}/vertices.txt", f"{root}/triangles.txt", f"{root}/uv_coords.txt",
                   f"{root}/uv_map2face.txt")

    '''
    # calling method

    blender --background --python prepare_init_files.py -- \
    --root "/home/pramish/Downloads/smpl male GS init/from blender" \
    --subdivision 1 \
    "/home/pramish/Downloads/smpl male GS init/SMPL_maya/basicModel_m_lbs_10_207_0_v1.0.2.fbx"
    '''

    # Paths to your point cloud files

    path_pc1 = '../../data/smpl/sd3/vertices.txt'  # Replace with your actual file path
    path_pc2 = '../../data/smpl/SMPL_MALE/v_template.txt'  # Replace with your actual file path

    # Load point clouds
    pc1 = load_point_cloud(path_pc1)
    pc2 = load_point_cloud(path_pc2)

    # Align pc1 to pc2
    aligned_pc1 = align_point_clouds(pc1, pc2)

    # Create the mapping index
    mapping_index = create_mapping_index(aligned_pc1, pc2)

    np.savetxt('../../data/smpl/sd3/mesh2smpl_idx.txt', mapping_index)
    # The mapping index has the shape (pc1_size, 1)
    print("Shape of mapping index:", mapping_index.shape)



