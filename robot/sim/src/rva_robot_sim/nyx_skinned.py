"""Nyx showing meshes whose vertices the simulation sets, as the animated people's.

gs-nyx-plugin 0.1.10 streams the poses of rigid bodies and the vertices of soft bodies (FEM, PBD,
MPM), but not the vertices of a rigid entity built with `enable_custom_vverts`: it shows such a
mesh still, in its rest pose. Nyx has what is needed, a `DynamicMesh` instance whose vertices are
read from a buffer it shares with torch. `install` patches the plugin so that each part of such a
mesh (one per material) becomes a textured `DynamicMesh` after the soft bodies' in that buffer,
and Genesis' custom vertices are copied into it at each render, GPU to GPU.

The patch replaces the exporter's `_build_scene` and wraps five other private methods of the
plugin, so the plugin's version is pinned until the change is made upstream
(github.com/Genesis-Embodied-AI/genesis-nyx).
"""

# ruff: noqa: SLF001  # Patching the plugin is reaching into its private members.

import os
from collections.abc import Callable
from typing import Final

import cv2
import genesis as gs
import gs_nyx.nyx_py_renderer as npr
import gs_nyx.nyx_py_sdk as nps
import numpy as np
import torch
from genesis.engine.entities import RigidEntity
from genesis.engine.entities.rigid_entity.rigid_geom import RigidVisGeom
from genesis.utils.misc import qd_to_torch
from gs_nyx_plugin.nyx_renderer import NyxPyRenderer
from gs_nyx_plugin.nyx_scene_exporter import NyxSceneExporter
from gs_nyx_plugin.nyx_scene_utils import is_exportable_entity, should_export_at_geom_level

_CUT_OFF: Final = 0.75
"""Texture opacity under which a pixel is not drawn, for the hair's and eyelashes' strands. At
0.5, the soft edge of the eyelashes' cards darkens the eyes."""

type _Pairs = list[tuple[object, int]]
"""The plugin's (entity, instance uuid) pairs, in the order the instances were built."""


def install() -> None:
    """Patch the Nyx plugin, before a scene with such meshes is built. Idempotent."""
    if getattr(NyxSceneExporter, "_rva_skinned", False):
        return
    setattr(NyxSceneExporter, "_rva_skinned", True)  # noqa: B010  # A marker ty would not accept
    setattr(NyxSceneExporter, "_build_scene", _build_scene)  # noqa: B010  # Nor this
    _wrap(NyxSceneExporter, "_build_instances", _build_instances)
    for name in ("_num_deform_verts", "_num_deform_tris", "_num_deform_objs"):
        _add_to_count(name)
    _wrap(NyxPyRenderer, "_build_reference_table", _build_reference_table)
    _wrap(NyxPyRenderer, "_import_interop_buffers", _import_interop_buffers)
    _wrap(NyxPyRenderer, "_upload_static_deformable_data", _upload_static_deformable_data)
    _wrap(NyxPyRenderer, "_update_geometry_tensors", _update_geometry_tensors)


def _skinned_entities(scene: gs.Scene) -> list[RigidEntity]:
    entities: list[RigidEntity] = []
    for entity in scene.rigid_solver.entities:
        if isinstance(entity, RigidEntity) and entity.morph.enable_custom_vverts:
            assert entity.surface.vis_mode == "visual", f"{entity.morph} is not shown as a mesh"
            entities.append(entity)
    return entities


def _skinned_vgeoms(scene: gs.Scene) -> list[RigidVisGeom]:
    return [vgeom for entity in _skinned_entities(scene) for vgeom in entity.vgeoms]


# Exporter: the instance the plugin builds for each skinned entity is emptied, and stays as the
# target of its vgeoms' rigid references; one DynamicMesh per vgeom follows every other instance.


def _build_scene(self: NyxSceneExporter) -> None:
    """The plugin's, with room for a DynamicMesh per skinned vgeom after its instances."""
    asset = self._scene_asset
    assert asset is not None, "created by the exporter's constructor"
    # Genesis poses the vgeoms once while building the rigid solver when an entity has custom
    # vertices, before the links have poses: articulated robots' would be exported as NaN.
    self._scene.rigid_solver.update_vgeoms()
    count = sum(
        entity.n_vgeoms if should_export_at_geom_level(entity) else 1
        for entity in self._scene.entities
        if is_exportable_entity(entity)
    )
    asset.instance_resize(count + len(_skinned_vgeoms(self._scene)))
    asset.camera_resize(len(self._cameras))
    asset.light_resize(len(self._lights))
    asset.env_map_resize(len(self._env_maps))
    asset.light_field_resize(len(self._light_fields))
    self._build_instances()
    self._build_cameras()
    self._build_lights()
    self._build_env_maps()
    self._build_light_fields()


def _build_instances(original: Callable[..., None], self: NyxSceneExporter) -> None:
    original(self)
    asset = self._scene_asset
    assert asset is not None, "created by the exporter's constructor"
    vgeoms = _skinned_vgeoms(self._scene)
    first = asset.instance_size() - len(vgeoms)
    entities = _skinned_entities(self._scene)
    emptied = {uuid for entity, uuid in self._entity_uuid_pairs if entity in entities}
    for index in range(first):
        instance = asset.get_instance(index)
        if instance.uuid in emptied:
            # Nyx draws disabled instances anyway: an empty box stands in for the GLB.
            instance.type = nps.EInstanceType.Primitive
            instance.primitive_type = nps.EPrimitiveType.Box
            instance.mesh_uri = ""
            instance.scale = nps.float3(0.0, 0.0, 0.0)
            asset.set_instance(index, instance)
    for offset, vgeom in enumerate(vgeoms):
        instance = asset.get_instance(first + offset)
        instance.type = nps.EInstanceType.DynamicMesh
        instance.dynamicMesh_numVertices = vgeom.n_vverts
        instance.dynamicMesh_numTriangles = vgeom.n_vfaces
        # Its vertices are in the world: Nyx's shaders turn Genesis' z-up into its y-up.
        instance.position = nps.float3(0.0, 0.0, 0.0)
        instance.rotation = nps.quaternion(0.0, 0.0, 0.0, 1.0)
        instance.scale = nps.float3(1.0, 1.0, 1.0)
        instance.matOverride = _material(vgeom, self._export_folder)
        instance.smooth = True
        instance.enabled = True
        instance.uuid = nps.generate_uuid()
        self._entity_uuid_pairs.append((vgeom.entity, instance.uuid))
        asset.set_instance(first + offset, instance)


def _material(vgeom: RigidVisGeom, folder: str) -> nps.MaterialAsset:
    """The vgeom's colour texture as a full material, cut out where its opacity is low.

    Genesis keeps a GLB texture's alpha apart, as the surface's opacity texture: it goes back
    into the PNG written for Nyx, which reads opacity from the colour texture's alpha.
    """
    surface = vgeom.vmesh.surface
    texture = surface.diffuse_texture
    assert isinstance(texture, gs.textures.ImageTexture), "the GLB is textured"
    bgr = cv2.cvtColor(np.asarray(texture.image_array), cv2.COLOR_RGB2BGR)
    opacity = surface.opacity_texture
    cut_out = isinstance(opacity, gs.textures.ImageTexture)
    image = np.dstack((bgr, np.asarray(opacity.image_array))) if cut_out else bgr
    path = os.path.join(folder, f"skinned_{vgeom.uid}.png")
    cv2.imwrite(path, image)
    material = nps.MaterialAsset()
    material.set_properties(nps.EMaterialProperty.AlbedoColor)
    material.albedoColor = nps.float4(1.0, 1.0, 1.0, 1.0)
    material.set_properties(nps.EMaterialProperty.AlbedoTexture)
    # Relative to the working directory, as the plugin writes the paths of Genesis' textures.
    material.albedoTexture = os.path.relpath(path, ".").replace("\\", "/")
    # The GLB's factors, which Genesis keeps as single-colour textures.
    roughness, metallic = surface.roughness_texture, surface.metallic_texture
    assert isinstance(roughness, gs.textures.ColorTexture), "the roughness is a factor"
    assert isinstance(metallic, gs.textures.ColorTexture), "the metalness is a factor"
    material.set_properties(nps.EMaterialProperty.PerceptualRoughness)
    material.perceptualRoughness = float(roughness.color[0])
    material.set_properties(nps.EMaterialProperty.Metalness)
    material.metalness = float(metallic.color[0])
    if cut_out:
        material.set_properties(nps.EMaterialProperty.Clipping)
        material.clipping = nps.EMaterialClipping.On
        material.set_properties(nps.EMaterialProperty.CutOffValue)
        material.cutOff = _CUT_OFF
        # Strands are cards seen from both sides.
        material.set_properties(nps.EMaterialProperty.Sidedness)
        material.sidedness = nps.EMaterialSidedness.Double
    return material


# Renderer: the skinned vertices and triangles are a section of the shared deformable buffers,
# after the soft bodies' (FEM, PBD, MPM, SPH), in the order of Genesis' custom vertices.


def _counts(renderer: NyxPyRenderer) -> dict[str, int]:
    vgeoms = _skinned_vgeoms(renderer._scene)
    return {
        "_num_deform_verts": sum(vgeom.n_vverts for vgeom in vgeoms),
        "_num_deform_tris": sum(vgeom.n_vfaces for vgeom in vgeoms),
        "_num_deform_objs": len(vgeoms),
    }


def _add_to_count(name: str) -> None:
    original = getattr(NyxPyRenderer, name).fget

    def count(self: NyxPyRenderer) -> int:
        return int(original(self)) + _counts(self)[name]

    setattr(NyxPyRenderer, name, property(count))


def _start(renderer: NyxPyRenderer, name: str) -> int:
    """Where the skinned section starts in the buffer that `name` counts."""
    return int(getattr(renderer, name)) - _counts(renderer)[name]


def _build_reference_table(
    original: Callable[..., None],
    self: NyxPyRenderer,
    update_desc: npr.BridgeUpdateDesc,
    pair_list: _Pairs,
) -> None:
    original(self, update_desc, pair_list)
    uuids = [
        uuid
        for entity in _skinned_entities(self._scene)
        # The entity's first instance is the emptied one; then its vgeoms', in order.
        for uuid in [uuid for paired, uuid in pair_list if paired is entity][1:]
    ]
    first = update_desc.numRigidGeom + _start(self, "_num_deform_objs")
    for offset, uuid in enumerate(uuids):
        reference = update_desc.get_reference(first + offset)
        reference.targetUUID = uuid
        reference.nodeName = ""
        update_desc.set_reference(first + offset, reference)


def _import_interop_buffers(original: Callable[..., None], self: NyxPyRenderer) -> None:
    original(self)
    start = _start(self, "_num_deform_verts")
    count = _counts(self)["_num_deform_verts"]
    assert count == self._rigid_solver.n_custom_vverts, "every custom vertex is a skinned one"
    _section[id(self)] = self._deformable_verts_backend[:, start : start + count]


def _upload_static_deformable_data(original: Callable[..., None], self: NyxPyRenderer) -> None:
    original(self)
    vertex = _start(self, "_num_deform_verts")
    triangle = _start(self, "_num_deform_tris")
    for vgeom in _skinned_vgeoms(self._scene):
        uvs = torch.as_tensor(np.asarray(vgeom.vmesh.uvs), dtype=torch.float32)
        self._deformable_uvs_backend[vertex : vertex + vgeom.n_vverts].copy_(uvs)
        faces = torch.as_tensor(vgeom.init_vfaces + vertex, dtype=torch.int32).reshape(-1)
        self._deformable_indices_backend[:, triangle * 3 : (triangle + vgeom.n_vfaces) * 3] = faces
        vertex += vgeom.n_vverts
        triangle += vgeom.n_vfaces


def _update_geometry_tensors(original: Callable[..., None], self: NyxPyRenderer) -> None:
    original(self)
    section = _section.get(id(self))
    if section is None:
        return
    # Genesis keeps every custom vertex in one buffer, entity after entity and vgeom after vgeom:
    # the section's order.
    vverts = qd_to_torch(self._rigid_solver.dyn_state.vverts.pos, transpose=True, copy=False)
    section.copy_(vverts[:, : section.shape[1]])


_section: Final[dict[int, torch.Tensor]] = {}
"""Each renderer's view of its skinned section of Nyx's vertex buffer, by the renderer's id."""


def _wrap(owner: type, name: str, wrapper: Callable[..., None]) -> None:
    original = getattr(owner, name)

    def wrapped(self: object, *arguments: object) -> None:
        wrapper(original, self, *arguments)

    setattr(owner, name, wrapped)
