import os


def patch():
    msvc = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130"
    sdk = r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0"
    sdk_lib = r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0"

    os.environ["INCLUDE"] = ";".join(
        [
            msvc + r"\include",
            sdk + r"\ucrt",
            sdk + r"\um",
            sdk + r"\shared",
        ]
    )

    os.environ["LIB"] = ";".join(
        [
            msvc + r"\lib\x64",
            sdk_lib + r"\ucrt\x64",
            sdk_lib + r"\um\x64",
        ]
    )
