import numpy as np

from medreg.batch import BatchItem, parameter_hash, run_batch
from medreg.image import MedicalImage, make_affine
from medreg.register import register_rigid
from medreg.synthetic import synthetic_pair
from medreg.transform import RigidTransform


def test_synthetic_rigid_registration_recovers_known_transform():
    fixed, moving, truth = synthetic_pair(
        shape=(32, 36, 30),
        rotation_angles_deg=(4.0, -5.0, 3.0),
        translation=(3.0, -2.0, 1.5),
    )
    report = run_batch(
        [(BatchItem("case", levels=(3, 2, 1), max_iterations=70), fixed, moving)],
        "/tmp/medreg-test-batch",
    )
    item = report["items"][0]
    assert item["result"]["status"] == "success"
    params = np.asarray(item["result"]["parameters_rotvec_xyz_translation_xyz"])
    estimated = RigidTransform.from_parameters(params)
    true_points = np.asarray([[0.0, 0.0, 0.0], [8.0, -6.0, 4.0], [-5.0, 3.0, 9.0]])
    assert np.allclose(estimated.apply(true_points), truth.apply(true_points), atol=2.0)
    assert item["result"]["coverage"]["coverage_fraction"] > 0.6
    assert item["result"]["error"]["ncc"] > 0.85
    assert item["effective_parameter_hash"]
    assert report["batch_parameter_hash"] == parameter_hash([item["effective_parameters"]])


def test_uninformative_images_report_failure_without_transformed_output():
    shape = (18, 18, 18)
    affine = make_affine(shape, spacing=(1.0, 1.0, 1.0))
    fixed = MedicalImage(np.zeros(shape, dtype=np.float32), affine)
    moving = MedicalImage(np.zeros(shape, dtype=np.float32), affine)
    result = register_rigid(fixed, moving, levels=(2, 1), max_iterations=10)
    assert result.status == "failed"
    assert result.converged is False
    assert result.transform is None
    assert result.transformed is None
    assert "did not meet acceptance" in result.message
