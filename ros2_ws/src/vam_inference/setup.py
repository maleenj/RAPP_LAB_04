from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vam_inference'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Maleen',
    maintainer_email='maleen@rapplab.org',
    description='ENACT Vision-Action Model inference for the TM12S robot',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'vam_tm12s_node = vam_inference.vam_tm12s_node:main',
            'vam_tm12s_node_viz = vam_inference.vam_tm12s_node_viz:main',
            'skeleton_relay = vam_inference.skeleton_relay:main',
            'servo_to_tm_pvt_bridge = vam_inference.servo_to_tm_pvt_bridge:main',
            # vam_pvt_streamer and vam_pvt_streamer_new2 are the SAME program:
            # the module keeps its historical filename, the clean name is the
            # documented command. Superseded variants live in legacy/experiments/.
            'vam_pvt_streamer = vam_inference.vam_pvt_streamer_new2:main',
            'vam_pvt_streamer_new2 = vam_inference.vam_pvt_streamer_new2:main',
        ],
    },
)
