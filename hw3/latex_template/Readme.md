ZigSkeleton

An extension of "ZigBear", a security research toolkit for Raspbee, nRF52840 and CC2531 radio modules. 

The Zigbear toolkit holds a custom proof of concept protocol based on IEEE 802.15.4 standard developed from the ground up. 

ZigSkeleton reduced the Zigbear codebase to only essential files and extended CLI to support a custom attack program. 

Attack chain includes: 
 - Off_On:
    - Targets a node to be taken "off" a network and then have it back "on"  or rejoin it's original network.
- NetworkRealignment:
    - A target node on an existing network is realigned via its network parameters to isolate it from its original network.
- Leave:
    - Allows a user to target a node to leave their network.
- PanIDConflict:
    - Allows a user to send a network address conflict frame to target network.
- AssociationFlooding:
    - Allows a user to flood a target network with 20 mock nodes.
- Blackhole:
    - Allows to reset a target Coordinator's networking parameters to eliminate 
    the routing path of its child nodes and thus cause their sent packets to be discarded. 
- Wormhole: 
    - 

Flashing the firmware
__
