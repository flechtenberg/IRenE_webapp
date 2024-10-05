// Add your custom JavaScript here

// Example: Display an alert when the form is submitted
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    if(form) {
        form.addEventListener('submit', function(event) {
            alert('Processing your request...');
        });
    }
});
