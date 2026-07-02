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
    description='Vision-Action Model inference node for UR10 robot mirroring',
    license='MIT',
    entry_points={
        'console_scripts': [
            'vam_inference_node = vam_inference.vam_node:main',
            'vam_tm12s_node = vam_inference.vam_tm12s_node:main',
            'vam_tm12s_node_viz = vam_inference.vam_tm12s_node_viz:main',
            'skeleton_relay = vam_inference.skeleton_relay:main',
            'servo_to_tm_vjog_bridge = vam_inference.servo_to_tm_vjog_bridge:main',
            'servo_to_tm_pvt_bridge = vam_inference.servo_to_tm_pvt_bridge:main',
            'servo_to_tm_ptp_bridge = vam_inference.servo_to_tm_ptp_bridge:main',
            'vam_pvt_streamer = vam_inference.vam_pvt_streamer:main',
            'vam_pvt_streamer_new = vam_inference.vam_pvt_streamer_new:main',
            'vam_pvt_streamer_new2 = vam_inference.vam_pvt_streamer_new2:main',
            'vam_pvt_streamer_new3 = vam_inference.vam_pvt_streamer_new3:main',
            'vam_pvt_streamer_new_fix = vam_inference.vam_pvt_streamer_new_fix:main',
            'vam_tm12s_node_diag = vam_inference.vam_tm12s_node_diag:main',
            'vam_pvt_streamer_diag = vam_inference.vam_pvt_streamer_diag:main',
        ],
    },
)
