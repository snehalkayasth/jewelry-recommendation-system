// Slider functionality
let currentSlide = 0;
const slides = document.querySelectorAll(".slide");

function changeSlide() {
    currentSlide++;
    if (currentSlide >= slides.length) {
        currentSlide = 0;
    }
    updateSlider();
}

function moveSlide(direction) {
    currentSlide += direction;
    if (currentSlide < 0) {
        currentSlide = slides.length - 1;
    } else if (currentSlide >= slides.length) {
        currentSlide = 0;
    }
    updateSlider();
}

function updateSlider() {
    const sliderContainer = document.querySelector(".slider-container");
    const offset = -currentSlide * 100;
    sliderContainer.style.transform = `translateX(${offset}%)`;
}

setInterval(changeSlide, 5000); // Change slide every 5 seconds
// guide animation==>
    // JavaScript for Guide Section
document.getElementById('start-guide-btn').addEventListener('click', function() {
    let stepIndex = 0;
    const steps = document.querySelectorAll('.step');
    const finalStep = document.getElementById('final-step');
    const interval = setInterval(function() {
        if (stepIndex < steps.length) {
            steps[stepIndex].classList.add('active');
            stepIndex++;
        } else {
            clearInterval(interval); // Stop the interval after all steps are revealed
            finalStep.classList.remove('hidden');
        }
    }, 1000); // Show each step after 1 second

    // Hide the "Start Guide" button
    this.style.display = 'none';
});
