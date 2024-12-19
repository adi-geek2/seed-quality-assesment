<?php
$uploadDir = __DIR__ . '/uploads/';
$allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (!empty($_FILES['seedImage']['name'])) {
        $fileName = basename($_FILES['seedImage']['name']);
        $filePath = $uploadDir . $fileName;
        $fileType = $_FILES['seedImage']['type'];

        if (in_array($fileType, $allowedTypes)) {
            if (!is_writable($uploadDir)) {
                die("Uploads directory is not writable.");
            }

            if (move_uploaded_file($_FILES['seedImage']['tmp_name'], $filePath)) {
                // Call the .bat file to execute the Python script
                $batFile = __DIR__ . '/run_project.bat';
                $pythonScript = __DIR__ . '/process_image.py';
                $command = escapeshellcmd("cmd /c \"$batFile $pythonScript $filePath\"");
                $output = shell_exec($command . " 2>&1"); // Execute and capture output

                // Log the output for debugging
                file_put_contents("debug.log", $output);

                // Parse JSON output
                $result = json_decode($output, true);
                if (isset($result['error'])) {
                    echo "Error: " . htmlspecialchars($result['error']);
                } elseif (isset($result['quality'])) {
                    $quality = $result['quality'];
                    header("Location: result.php?quality=" . urlencode($quality));
                    exit;
                } else {
                    echo "Error processing the image. Output: $output";
                }
            } else {
                echo "Failed to upload the image.";
            }
        } else {
            echo "Invalid file type. Only JPEG and PNG are allowed.";
        }
    } else {
        echo "Please select an image to upload.";
    }
} else {
    echo "Invalid request method.";
}
?>
