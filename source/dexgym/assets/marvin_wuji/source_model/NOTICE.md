# Marvin/Wuji simulation source used by DexGym

`dual_arm_marvin.xml` and the 70 meshes in `assets/` are copied byte-for-byte
from the pinned `FEGkkd/whole-body-teleop` source at commit
`5d4c345842f197bffeae7d7e08c4c3718230f9e0`. The XML SHA256 is
`6e0d569ec28c8ac26a84bfc9a48652b7a11a2f113193c2d3ea4191621280b3ce`.
Its first line records the local simulation conversion of `dual_arm.xml` and
the Marvin wrist/TCP convention. This copy is the input to the packaged USD
and the 54 DoF MuJoCo/Newton validation; DexGym does not edit the source
geometry or joint definitions.

The Marvin vendor notice is reproduced in `LICENSE.Marvin` (Apache 2.0,
copyright 2025 上海孚晞科技有限公司). The WujiHand2 Beta 2 notice is reproduced in
`LICENSE.Wuji` (MIT, copyright 2025 Wuji Technology). The combined MJCF and
meshes retain their upstream provenance. These notices do not replace any
conditions attached to upstream component files.
