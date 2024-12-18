<?php
// Configure upload directory and allowed file types
$uploadDir = 'uploads/';
$allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];

// Check if a file is uploaded
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (!empty($_FILES['seedImage']['name'])) {
        $fileName = basename($_FILES['seedImage']['name']);
        $filePath = $uploadDir . $fileName;
        $fileType = $_FILES['seedImage']['type'];

        // Validate file type
        if (in_array($fileType, $allowedTypes)) {
            // Move the uploaded file to the target directory
            if (move_uploaded_file($_FILES['seedImage']['tmp_name'], $filePath)) {
                // Call Python script for image processing
                $pythonScript = 'python3 process_image.py';
                $command = escapeshellcmd($pythonScript . ' ' . $filePath);
                $output = shell_exec($command);

                // Parse the output and redirect to results page
                $result = json_decode($output, true); // Assuming JSON output from Python
                if ($result) {
                    header("Location: result.php?quality=" . urlencode($result['quality']) . "&defects=" . urlencode($result['defects']));
                    exit;
                } else {
                    echo "Error processing the image.";
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
