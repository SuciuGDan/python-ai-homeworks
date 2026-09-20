import pytest
import json
from hw_01_Validator.hw_01_StudentValidator import StudentValidator
from pydantic import ValidationError

with open("tests/student.json", "r") as f:
    fileStudentData = json.load(f)

def test_StudentValidator():
        try:
            student = StudentValidator.model_validate(fileStudentData, strict=True)
            print("Valid:", student)
        except ValidationError as e:
            print("Invalid data:", e)


def test_StudentValidator2():
    student = StudentValidator.model_validate(fileStudentData, strict=True)
    assert isinstance(student,StudentValidator)


@pytest.mark.parametrize("varsta",[-20,-150,555])
def test_StudentValidator3(varsta):
    data = {**fileStudentData, "varsta": varsta}
    with pytest.raises(ValidationError):
        StudentValidator.model_validate(data, strict=True)
